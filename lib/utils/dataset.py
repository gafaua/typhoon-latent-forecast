import numpy as np
import torch
from pyphoon2.DigitalTyphoonDataset import DigitalTyphoonDataset
from torch.utils.data import Dataset


class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        data = self.subset[index]
        if self.transform:
            return self.transform(data)
        return data

    def __len__(self):
        return len(self.subset)


class TemporalSequencePairDataset(Dataset):
    def __init__(self, images, labels, transform) -> None:
        self.images = images
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        size = len(self.images)
        #index2 = int(np.random.uniform() * np.min([6, (size - index)])) + index
        index2 = int(np.random.beta(2, 5) * np.min([6, (size - index)])) + index

        x1, lbl1 = self.images[index], self.labels[index]
        x2, lbl2 = self.images[index2], self.labels[index2]

        img1, dt1 = self.transform((x1, lbl1))
        img2, dt2 = self.transform((x2, lbl2))

        return img1, img2, (dt2-dt1).total_seconds()/3600

    def __len__(self):
        return len(self.images)



class NestedDigitalTyphoonDataset(DigitalTyphoonDataset):
    def __init__(self,
                 image_dir: str,
                 metadata_dir: str,
                 metadata_json: str,
                 labels,
                 split_dataset_by="image",
                 spectrum="Infrared",
                 load_data_into_memory=False,
                 ignore_list=None,
                 filter_func=None,
                 transform_func=None,
                 transform=None,
                 verbose=False) -> None:
        super().__init__(image_dir, metadata_dir, metadata_json, labels, split_dataset_by, spectrum, True, load_data_into_memory, ignore_list, filter_func, transform_func, transform, verbose)

    def __getitem__(self, idx):
        seq = self.get_ith_sequence(idx)
        images = seq.get_all_images_in_sequence()
        image_arrays = np.array([image.image() for image in images])
        labels = np.array([self._labels_from_label_strs(image, self.labels) for image in images])

        return TemporalSequencePairDataset(image_arrays, labels, self.transform)


class MoCoSequenceDataset(DigitalTyphoonDataset):
    def __init__(self,
                 image_dir: str,
                 metadata_dir: str,
                 metadata_json: str,
                 labels,
                 window_size_scheduler,
                 split_dataset_by="image",
                 spectrum="Infrared",
                 load_data_into_memory=False,
                 ignore_list=None,
                 filter_func=None,
                 transform_func=None,
                 transform=None,
                 verbose=False) -> None:
        super().__init__(image_dir,
                         metadata_dir,
                         metadata_json,
                         labels,
                         split_dataset_by,
                         spectrum,
                         True,
                         load_data_into_memory,
                         ignore_list,
                         filter_func,
                         transform_func,
                         transform,
                         verbose)
        self.window_size_scheduler = window_size_scheduler

    def __getitem__(self, idx):
        seq = self.get_ith_sequence(idx)
        images = seq.get_all_images_in_sequence()

        max_range = self.window_size_scheduler.get_value()

        idx1 = np.random.randint(0, len(images))
        idx2 = np.min([idx1 + np.random.randint(0, max_range), len(images) - 1])
        #print(idx1, idx2)

        return self.transform(images[idx1].image()), self.transform(images[idx2].image())


LABEL_SIZE = dict(
    month=12,
    day=31,
    hour=24,
    # grade=6,
)

NORMALIZATION = dict(
    pressure=[983.8, 22.5],
    wind=[36.7, 32.7],
    lat=[22.58, 10.6],
    lng=[136.2, 17.3],
)

class SequenceTyphoonDataset(DigitalTyphoonDataset):
    def __init__(self,
                 labels,
                 x,
                 y,
                 num_inputs,
                 num_preds,
                 interval=1,
                 output_all=False,
                 preprocessed_path=None,
                 latent_dim=None,
                 pred_diff=False,
                 prefix="/fs9/datasets/typhoon-202404/wnp",
                 spectrum="Infrared",
                 load_data_into_memory=False,
                 ignore_list=None,
                 filter_func=None,
                 transform_func=None,
                 transform=None,
                 verbose=False) -> None:
        """
        labels: labels to include in ["year", "month", "day", "hour", "grade", "lat", "lng", "pressure", "wind"]
        include_images: boolean to include or not images when generating sequences
        x: sequence data to use as inputs. should be array indices corresponding to the order in which labels are requested
        y: sequence data to use as targets. should be array indices corresponding to the order in which labels are requested
            images, if included are always included
        num_inputs: length of sequence used as input to the model
        num_preds: length of predicted datapoints
        preprocess: preprocess images to a smaller feature vector
        """
        super().__init__(f"{prefix}/image/",
                        f"{prefix}/metadata/",
                        f"{prefix}/metadata.json",
                         labels,
                         "sequence",
                         spectrum,
                         True,
                         load_data_into_memory,
                         ignore_list,
                         filter_func,
                         transform_func,
                         transform,
                         verbose)
        idx = 0
        self.x = []
        self.y = []

        for i, label in enumerate(labels):
            sz = LABEL_SIZE[label] if label in LABEL_SIZE else 1
            if i in x:
                self.x.extend(list(range(idx, idx+sz)))
            if i in y:
                self.y.extend(list(range(idx, idx+sz)))
            idx += sz

        if preprocessed_path is None:
            print("WARNING: no images used")
        else:
            assert latent_dim is not None
            if -1 in x:
                self.x.extend(list(range(idx, idx+latent_dim)))
            if -1 in y:
                self.y.extend(list(range(idx, idx+latent_dim)))

        print(f"\nExpected model inputs: {self.get_input_size()}")
        print(f"Expected model outputs: {self.get_output_size()}\n")

        self.preprocessed_path = preprocessed_path

        self.num_inputs = num_inputs
        self.num_preds = num_preds
        self.interval = interval
        self.output_all = output_all
        self.pred_diff = pred_diff

        self.slice_inputs = lambda start_idx: slice(start_idx, start_idx+(self.num_inputs*self.interval),self.interval)
        self.slice_outputs = lambda start_idx: slice(start_idx+(self.num_inputs*self.interval),start_idx+((self.num_inputs+self.num_preds)*self.interval), self.interval)

        # Post process sequences filter out too short sequences
        def filter_sequences(sequence):
            if sequence.get_num_images() < (self.num_inputs + self.num_preds)*self.interval+1 or sequence.images[0].year() < 1987:
                return True

            if "grade" in self.labels:
                # Filter out sequences which do not transition to ETS
                for image in sequence.images:
                    if image.grade() == 6:
                        return False
                return True
            else:
                return False

        for seq in self.sequences:
            if filter_sequences(seq):
                self.number_of_images -= seq.get_num_images()
                seq.images.clear()
                self.number_of_nonempty_sequences -= 1

        self.number_of_nonempty_sequences += 1


    def __getitem__(self, idx):
        seq = self.get_ith_sequence(idx)
        images = seq.get_all_images_in_sequence()

        labels = torch.stack([self._labels_from_label_strs(image, self.labels) for image in images])

        if self.preprocessed_path is not None:
            npz = np.load(f"{self.preprocessed_path}/{seq.sequence_str}.npz")
            names_to_features = dict(zip(npz["arr_1"], npz["arr_0"]))
            features = [names_to_features[str(img.image_filepath).split("/")[-1].split(".")[0]]
                        for img in images]
            features = torch.from_numpy(np.array(features))

            labels = torch.cat((labels, features), dim=1)

        if self.output_all:
            return labels, seq.sequence_str

        start_idx = np.random.randint(0, seq.get_num_images()-(self.num_inputs + self.num_preds)*self.interval)
        lab_inputs = labels[self.slice_inputs(start_idx), self.x]
        lab_preds = labels[self.slice_outputs(start_idx), self.y]

        if self.pred_diff:
            lab_preds = lab_preds - labels[self.slice_inputs(start_idx), self.y][-1]

        return lab_inputs, lab_preds

    def get_input_size(self):
        return len(self.x)

    def get_output_size(self):
        return len(self.y)

    def _labels_from_label_strs(self, image, label_strs):
        """
        Given an image and the label/labels to retrieve from the image, returns a single label or
        a list of labels

        :param image: image to access labels for
        :param label_strs: either a List of label strings or a single label string
        :return: a List of label strings or a single label string
        """
        if isinstance(label_strs, list) or isinstance(label_strs, tuple):
            label_ray = torch.cat([self._prepare_labels(image.value_from_string(label), label) for label in label_strs])
            return label_ray
        else:
            label = self._prepare_labels(image.value_from_string(label_strs), label_strs)
            return label

    def _prepare_labels(self, value, label):
        if label in LABEL_SIZE:
            one_hot = torch.zeros(LABEL_SIZE[label])
            if label == "hour":
                one_hot[value] = 1
            elif label == "grade":
                one_hot[value-2] = 1
            else:
                one_hot[value-1] = 1
            return one_hot
        else:
            # Normalize
            if label in NORMALIZATION:
                #print(label, value)
                mean, std = NORMALIZATION[label]
                return (torch.Tensor([value]) - mean) / std

            if label == "grade":
                return torch.Tensor([float(value)])

            return torch.Tensor([value])

    def get_sequence_images(self, seq_str):
        def crop(img, cropx=224, cropy=224):
            y,x = img.shape
            startx = x//2-(cropx//2)
            starty = y//2-(cropy//2)
            return img[starty:starty+cropy,startx:startx+cropx]
        idx = self._sequence_str_to_seq_idx[seq_str]
        seq = self.sequences[idx]
        images = seq.get_all_images_in_sequence()
        return [crop(image.image()) for image in images]

    def get_sequence(self, seq_str):
        idx = self._sequence_str_to_seq_idx[seq_str]
        return self.__getitem__(idx)


