import os
import torch
import numpy as np
from PIL import Image
from termcolor import cprint

from preprocess import clean_text
from base_dataset import BaseDataset, expand2square

# from paths import dataroot
import torchvision.transforms as transforms

from transformers import ElectraTokenizer, CLIPTokenizer, AutoTokenizer


# -----------------------------------------------------
# Label dictionaries for all 3 tasks
# -----------------------------------------------------
labels_task1 = {
    'informative': 1,
    'not_informative': 0
}

labels_task2_full = {
    'infrastructure_and_utility_damage': 0,
    'not_humanitarian': 1,
    'other_relevant_information': 2,
    'rescue_volunteering_or_donation_effort': 3,
    'vehicle_damage': 4,
    'affected_individuals': 5,
    'injured_or_dead_people': 6,
    'missing_or_found_people': 7,
}

labels_task2 = {
    'infrastructure_and_utility_damage': 0,
    'not_humanitarian': 1,
    'other_relevant_information': 2,
    'rescue_volunteering_or_donation_effort': 3,
    'vehicle_damage': 4,
    'affected_individuals': 5,
    'injured_or_dead_people': 5,
    'missing_or_found_people': 5,
}

labels_task3 = {
    'little_or_no_damage': 0,
    'mild_damage': 1,
    'severe_damage': 2
}


# -----------------------------------------------------
# Safe label mapper
# -----------------------------------------------------
def map_label(label_str, mapping):
    """Convert textual label → int; missing labels map to -1."""
    if label_str is None:
        return -1
    s = str(label_str).strip()
    if s == "" or s.lower() == "nan":
        return -1
    return mapping.get(s, -1)


# -----------------------------------------------------
# Main Dataset Class
# -----------------------------------------------------
class CrisisMMDataset(BaseDataset):

    # -------------------------------------------------
    # Tokenizers
    # -------------------------------------------------
    def _init_tokenizers(self):
        self.electra_tokenizer = ElectraTokenizer.from_pretrained(
            'google/electra-base-discriminator'
        )

        self.clip_tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

        # LLaMA tokenizer for possible extension
        llama_name = "meta-llama/Llama-3.2-1B"
        self.llama_tokenizer = AutoTokenizer.from_pretrained(llama_name)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

    # -------------------------------------------------
    def tokenize_clip(self, sentence):
        ids = self.clip_tokenizer(
            clean_text(sentence),
            padding='max_length',
            max_length=77,
            truncation=True
        )
        return {k: torch.tensor(v) for k, v in ids.items()}

    # -------------------------------------------------
    def tokenize_llama(self, sentence):
        ids = self.llama_tokenizer(
            clean_text(sentence),
            padding='max_length',
            max_length=128,
            truncation=True
        )
        return {k: torch.tensor(v) for k, v in ids.items()}

    # -------------------------------------------------
    # Read merged TSV
    # -------------------------------------------------
    def read_merged_tsv(self, ann_file):
        with open(ann_file, encoding="utf-8") as f:
            lines = [l.rstrip("\n") for l in f.readlines()]

        if not lines:
            self.data_list = []
            return

        header = lines[0].split("\t")
        header_idx = {h.strip().lower(): i for i, h in enumerate(header)}

        def get(parts, key):
            k = key.lower()
            if k in header_idx and header_idx[k] < len(parts):
                return parts[header_idx[k]]
            return ""

        data_list = []

        for line in lines[1:]:
            if not line:
                continue
            parts = line.split("\t")

            tweet_text = get(parts, "tweet_text") or get(parts, "text")
            llava_text = get(parts, "llava_text")

            if self.opt.text_from == "llava" and llava_text:
                caption = tweet_text + " " + llava_text
            else:
                caption = tweet_text

            # Labels
            l1 = map_label(get(parts, "label_task1"), labels_task1)
            l2 = map_label(get(parts, "label_task2"), labels_task2)
            l3 = map_label(get(parts, "label_task3"), labels_task3)

            # Image path (relative path is assumed)
            image_rel = get(parts, "image")
            path_image = (
                f"{self.dataset_root}/{image_rel}" if image_rel else ""
            )

            entry = {
                "path_image": path_image,
                "tweet_id": get(parts, "tweet_id"),
                "image_id": get(parts, "image_id"),
                "text": caption,
                "text_tokens": self.tokenize_clip(caption),
                "llama_tokens": self.tokenize_llama(caption),
                "label_task1": l1,
                "label_task2": l2,
                "label_task3": l3,
                "event": get(parts, "event_category") or get(parts, "event")
            }

            data_list.append(entry)

        self.data_list = data_list

    # -------------------------------------------------
    # Initialization
    # -------------------------------------------------
    def initialize(self, opt, phase="train", shuffle=False):
        self.opt = opt
        self.mode = phase
        self.shuffle = shuffle

        # root folder
        self.dataset_root = "./datasets/CrisisMMD_v2.0"
        self.image_root = f"{self.dataset_root}/data_image"

        # initialize tokenizers
        self._init_tokenizers()

        # -------------------------------------------------
        # Annotation file (always merged TSV)
        # -------------------------------------------------
        merged_dir = "./merged"

        ann_file = os.path.join(merged_dir, f"merged_{phase}.tsv")

        # Optional safety check
        if not os.path.isfile(ann_file):
            raise FileNotFoundError(
                f"Expected merged TSV file not found: {ann_file}"
            )

        # read file
        self.read_merged_tsv(ann_file)

        # shuffle
        if shuffle:
            np.random.default_rng(seed=0).shuffle(self.data_list)

        # limit size
        max_size = getattr(opt, "max_dataset_size", len(self.data_list))
        self.data_list = self.data_list[:max_size]

        cprint(f"[*] {len(self.data_list)} samples loaded.", "yellow")

        self.N = len(self.data_list)

        # image transforms
        load_size = getattr(opt, "load_size", 224)
        crop_size = getattr(opt, "crop_size", 224)

        self.transforms = transforms.Compose([
            transforms.Lambda(lambda img: expand2square(img)),
            transforms.Resize((load_size, load_size)),
            transforms.RandomHorizontalFlip(0.2),
            transforms.RandomCrop((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5)),
        ])

        self.test_transforms = transforms.Compose([
            transforms.Lambda(lambda img: expand2square(img)),
            transforms.Resize((load_size, load_size)),
            transforms.RandomCrop((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5)),
        ])

    # -------------------------------------------------
    def __getitem__(self, index):
        item = dict(self.data_list[index])  # shallow copy

        img_path = item["path_image"]
        load_size = getattr(self.opt, "load_size", 224)

        if img_path and os.path.exists(img_path):
            with Image.open(img_path).convert("RGB") as img:
                if self.mode == "test":
                    image = self.test_transforms(img)
                else:
                    image = self.transforms(img)
        else:
            image = torch.zeros(3, load_size, load_size)

        item["image"] = image
        return item

    # -------------------------------------------------
    def __len__(self):
        return len(self.data_list)

    def name(self):
        return "CrisisMMDataset"


if __name__ == "__main__":
    pass
