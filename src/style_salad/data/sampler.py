import random
from collections import defaultdict

from torch.utils.data import BatchSampler


class TrainSampler(BatchSampler):
    """Build style-balanced batches from Dataset100STYLE items."""

    def __init__(self, config, dataset):
        self.dataset = dataset
        self.batch_size = int(config["batch_size"])
        self.samples_per_style = int(config["samples_per_style"])

        self.styles_to_indices = defaultdict(list)
        for idx, item in enumerate(dataset.items):
            self.styles_to_indices[item["style_idx"]].append(idx)

        self.styles = [style for style, indices in self.styles_to_indices.items() if indices]
        if not self.styles:
            raise ValueError("TrainSampler found no styles with samples in the dataset.")

        self._generate_batches()

    def _generate_batches(self):
        self.batches = []
        styles_per_batch = max(1, self.batch_size // self.samples_per_style)
        num_batches = len(self.dataset) // self.batch_size
        all_indices = [idx for style in self.styles for idx in self.styles_to_indices[style]]

        for _ in range(num_batches):
            if styles_per_batch <= len(self.styles):
                selected_styles = random.sample(self.styles, styles_per_batch)
            else:
                selected_styles = random.choices(self.styles, k=styles_per_batch)

            batch = []
            for style in selected_styles:
                pool = self.styles_to_indices[style]
                if len(pool) >= self.samples_per_style:
                    batch.extend(random.sample(pool, self.samples_per_style))
                else:
                    batch.extend(random.choices(pool, k=self.samples_per_style))

            if len(batch) > self.batch_size:
                batch = batch[:self.batch_size]
            elif len(batch) < self.batch_size:
                batch.extend(random.choices(all_indices, k=self.batch_size - len(batch)))
            self.batches.append(batch)

    def __iter__(self):
        self._generate_batches()
        random.shuffle(self.batches)
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


SAMPLER_REGISTRY = {"TrainSampler": TrainSampler}
