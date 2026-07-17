
import argparse
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd


class FastDatasetSplitOptimizer:
    """
    Fast image-level 80/20 split optimizer for object detection datasets.

    Features:
    - Reads only GT.csv, not image files.
    - Keeps all annotations from one image in the same subset.
    - Forces classes occurring in one image to remain in train.
    - Ensures every class occurring in at least two images appears in both train and test.
    - Balances per-class image counts and instance counts.
    - Uses greedy construction plus lightweight local swaps.
    """

    def __init__(
        self,
        gt_csv: Path,
        output_dir: Path,
        filename_column: str = "filename",
        class_column: str = "Class ID",
        train_ratio: float = 0.8,
        seed: int = 42,
        restarts: int = 20,
        swap_rounds: int = 3000,
        candidate_pool: int = 40,
    ) -> None:
        self.gt_csv = Path(gt_csv)
        self.output_dir = Path(output_dir)
        self.filename_column = filename_column
        self.class_column = class_column
        self.train_ratio = train_ratio
        self.test_ratio = 1.0 - train_ratio
        self.seed = seed
        self.restarts = restarts
        self.swap_rounds = swap_rounds
        self.candidate_pool = candidate_pool

        self.df = None
        self.all_images: List[str] = []
        self.all_classes: List = []

        self.image_to_classes: Dict[str, Set] = {}
        self.image_to_instance_counts: Dict[str, Dict] = {}
        self.class_to_images: Dict[object, Set[str]] = defaultdict(set)

        self.class_total_images: Dict[object, int] = {}
        self.class_total_instances: Dict[object, int] = {}

        self.total_instances = 0
        self.target_test_size = 0

        self.unsplittable_classes: List = []
        self.splittable_classes: List = []
        self.forced_train_images: Set[str] = set()

        self.best_test_set: Set[str] | None = None
        self.best_score = math.inf

    def load_data(self) -> None:
        if not self.gt_csv.exists():
            raise FileNotFoundError(f"找不到 GT.csv：{self.gt_csv.resolve()}")

        self.df = pd.read_csv(self.gt_csv)

        required = {self.filename_column, self.class_column}
        missing = required - set(self.df.columns)

        if missing:
            raise ValueError(
                f"GT.csv 缺少字段：{sorted(missing)}；"
                f"当前字段：{self.df.columns.tolist()}"
            )

        self.df = self.df.dropna(
            subset=[self.filename_column, self.class_column]
        ).copy()

        self.df[self.filename_column] = (
            self.df[self.filename_column]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        self.all_images = sorted(
            self.df[self.filename_column].unique().tolist()
        )

        self.all_classes = sorted(
            self.df[self.class_column].unique().tolist(),
            key=lambda x: str(x),
        )

        self.total_instances = len(self.df)
        self.target_test_size = round(
            len(self.all_images) * self.test_ratio
        )

        self.image_to_classes = (
            self.df.groupby(self.filename_column)[self.class_column]
            .apply(set)
            .to_dict()
        )

        for filename, group in self.df.groupby(self.filename_column):
            self.image_to_instance_counts[filename] = (
                group[self.class_column]
                .value_counts()
                .to_dict()
            )

        for filename, classes in self.image_to_classes.items():
            for class_id in classes:
                self.class_to_images[class_id].add(filename)

        self.class_total_images = {
            class_id: len(self.class_to_images[class_id])
            for class_id in self.all_classes
        }

        self.class_total_instances = (
            self.df[self.class_column]
            .value_counts()
            .to_dict()
        )

        self.unsplittable_classes = [
            class_id
            for class_id in self.all_classes
            if self.class_total_images[class_id] < 2
        ]

        self.splittable_classes = [
            class_id
            for class_id in self.all_classes
            if self.class_total_images[class_id] >= 2
        ]

        for class_id in self.unsplittable_classes:
            self.forced_train_images.update(
                self.class_to_images[class_id]
            )

        print("===== Dataset summary =====")
        print(f"Images: {len(self.all_images)}")
        print(f"Classes: {len(self.all_classes)}")
        print(f"Instances: {self.total_instances}")
        print(
            f"Target train/test: "
            f"{len(self.all_images) - self.target_test_size}/"
            f"{self.target_test_size}"
        )
        print(f"Splittable classes: {len(self.splittable_classes)}")
        print(f"Unsplittable classes: {len(self.unsplittable_classes)}")
        print(f"Forced-train images: {len(self.forced_train_images)}")

    def get_test_class_image_counts(
        self,
        test_set: Set[str],
    ) -> Dict[object, int]:
        counts = defaultdict(int)

        for filename in test_set:
            for class_id in self.image_to_classes[filename]:
                counts[class_id] += 1

        return counts

    def get_test_class_instance_counts(
        self,
        test_set: Set[str],
    ) -> Dict[object, int]:
        counts = defaultdict(int)

        for filename in test_set:
            for class_id, count in (
                self.image_to_instance_counts[filename].items()
            ):
                counts[class_id] += count

        return counts

    def score(self, test_set: Set[str]) -> float:
        if len(test_set) != self.target_test_size:
            return 1e9

        if test_set & self.forced_train_images:
            return 1e9

        test_image_counts = self.get_test_class_image_counts(test_set)
        test_instance_counts = self.get_test_class_instance_counts(
            test_set
        )

        image_errors = []
        instance_errors = []
        penalty = 0.0

        for class_id in self.all_classes:
            total_images = self.class_total_images[class_id]
            total_instances = self.class_total_instances[class_id]

            test_images = test_image_counts.get(class_id, 0)
            train_images = total_images - test_images
            test_instances = test_instance_counts.get(class_id, 0)

            if total_images >= 2:
                if test_images == 0 or train_images == 0:
                    penalty += 10000.0
            elif test_images > 0:
                penalty += 10000.0

            image_ratio = test_images / total_images
            instance_ratio = test_instances / total_instances

            rarity_weight = 1.0 / math.sqrt(total_images)

            image_errors.append(
                rarity_weight * abs(image_ratio - self.test_ratio)
            )
            instance_errors.append(
                rarity_weight * abs(instance_ratio - self.test_ratio)
            )

        total_test_instances = sum(
            sum(self.image_to_instance_counts[img].values())
            for img in test_set
        )

        total_instance_ratio = (
            total_test_instances / self.total_instances
        )

        return (
            penalty
            + 0.60 * float(np.mean(image_errors))
            + 0.30 * float(np.mean(instance_errors))
            + 0.10 * abs(
                total_instance_ratio - self.test_ratio
            )
        )

    def can_add(
        self,
        filename: str,
        test_set: Set[str],
        current_counts: Dict[object, int],
    ) -> bool:
        if filename in self.forced_train_images:
            return False

        for class_id in self.image_to_classes[filename]:
            total_images = self.class_total_images[class_id]

            if total_images < 2:
                return False

            if current_counts.get(class_id, 0) + 1 >= total_images:
                return False

        return True

    def greedy_initial_split(
        self,
        rng: random.Random,
    ) -> Set[str] | None:
        test_set: Set[str] = set()
        test_class_counts = defaultdict(int)

        rare_first = sorted(
            self.splittable_classes,
            key=lambda class_id: (
                self.class_total_images[class_id],
                self.class_total_instances[class_id],
                str(class_id),
            ),
        )

        # First guarantee test coverage for all splittable classes.
        for class_id in rare_first:
            if test_class_counts.get(class_id, 0) > 0:
                continue

            candidates = [
                filename
                for filename in self.class_to_images[class_id]
                if filename not in test_set
                and self.can_add(
                    filename,
                    test_set,
                    test_class_counts,
                )
            ]

            if not candidates:
                return None

            rng.shuffle(candidates)

            best_value = -math.inf
            best_candidates = []

            for filename in candidates:
                coverage_gain = 0.0

                for contained_class in self.image_to_classes[filename]:
                    if (
                        self.class_total_images[contained_class] >= 2
                        and test_class_counts.get(
                            contained_class, 0
                        ) == 0
                    ):
                        coverage_gain += (
                            1.0
                            / math.sqrt(
                                self.class_total_images[
                                    contained_class
                                ]
                            )
                        )

                target_instances_per_test_image = (
                    self.total_instances
                    * self.test_ratio
                    / self.target_test_size
                )

                image_instances = sum(
                    self.image_to_instance_counts[filename].values()
                )

                density_penalty = (
                    abs(
                        image_instances
                        - target_instances_per_test_image
                    )
                    * 0.001
                )

                value = coverage_gain - density_penalty

                if value > best_value + 1e-12:
                    best_value = value
                    best_candidates = [filename]
                elif abs(value - best_value) <= 1e-12:
                    best_candidates.append(filename)

            chosen = rng.choice(best_candidates)
            test_set.add(chosen)

            for contained_class in self.image_to_classes[chosen]:
                test_class_counts[contained_class] += 1

            if len(test_set) > self.target_test_size:
                return None

        # Then fill remaining positions using deficit-based greedy selection.
        while len(test_set) < self.target_test_size:
            candidates = [
                filename
                for filename in self.all_images
                if filename not in test_set
                and self.can_add(
                    filename,
                    test_set,
                    test_class_counts,
                )
            ]

            if not candidates:
                return None

            rng.shuffle(candidates)

            if len(candidates) > 200:
                candidates = candidates[:200]

            best_value = -math.inf
            best_candidates = []

            current_instance_counts = (
                self.get_test_class_instance_counts(test_set)
            )

            for filename in candidates:
                value = 0.0

                for class_id in self.image_to_classes[filename]:
                    total_images = self.class_total_images[class_id]
                    current_images = test_class_counts.get(
                        class_id, 0
                    )
                    target_images = total_images * self.test_ratio

                    image_deficit = target_images - current_images
                    value += image_deficit / math.sqrt(total_images)

                    total_instances = (
                        self.class_total_instances[class_id]
                    )
                    current_instances = (
                        current_instance_counts.get(class_id, 0)
                    )
                    add_instances = (
                        self.image_to_instance_counts[filename]
                        .get(class_id, 0)
                    )
                    target_instances = (
                        total_instances * self.test_ratio
                    )

                    before_error = abs(
                        current_instances - target_instances
                    )
                    after_error = abs(
                        current_instances
                        + add_instances
                        - target_instances
                    )

                    value += (
                        before_error - after_error
                    ) / math.sqrt(total_instances)

                if value > best_value + 1e-12:
                    best_value = value
                    best_candidates = [filename]
                elif abs(value - best_value) <= 1e-12:
                    best_candidates.append(filename)

            chosen = rng.choice(best_candidates)
            test_set.add(chosen)

            for class_id in self.image_to_classes[chosen]:
                test_class_counts[class_id] += 1

        return test_set

    def fast_local_search(
        self,
        test_set: Set[str],
        rng: random.Random,
        restart_index: int,
    ) -> Tuple[Set[str], float]:
        current = set(test_set)
        current_score = self.score(current)

        all_image_set = set(self.all_images)

        start_time = time.time()
        last_improvement_round = 0
        improvement_count = 0

        # About 20 progress updates per restart.
        progress_interval = max(1, self.swap_rounds // 20)

        for round_index in range(1, self.swap_rounds + 1):
            train_set = (
                all_image_set
                - current
                - self.forced_train_images
            )

            if not train_set:
                print(
                    f"  Restart {restart_index}/{self.restarts}: "
                    "no train candidates remain; stopping."
                )
                break

            test_candidates = rng.sample(
                list(current),
                min(self.candidate_pool, len(current)),
            )

            train_candidates = rng.sample(
                list(train_set),
                min(self.candidate_pool, len(train_set)),
            )

            best_swap = None
            best_swap_score = current_score

            for out_image in test_candidates:
                for in_image in train_candidates:
                    candidate = (
                        current - {out_image}
                    ) | {in_image}

                    candidate_score = self.score(candidate)

                    if candidate_score < best_swap_score - 1e-12:
                        best_swap_score = candidate_score
                        best_swap = (out_image, in_image)

            if best_swap is not None:
                out_image, in_image = best_swap
                current.remove(out_image)
                current.add(in_image)
                current_score = best_swap_score

                last_improvement_round = round_index
                improvement_count += 1

            if (
                round_index == 1
                or round_index % progress_interval == 0
                or round_index == self.swap_rounds
            ):
                elapsed = time.time() - start_time
                progress = round_index / self.swap_rounds

                if progress > 0:
                    estimated_total = elapsed / progress
                    remaining = max(0.0, estimated_total - elapsed)
                else:
                    remaining = 0.0

                no_improve = round_index - last_improvement_round

                print(
                    f"  Restart {restart_index}/{self.restarts} | "
                    f"swap {round_index}/{self.swap_rounds} "
                    f"({progress:6.1%}) | "
                    f"score={current_score:.10f} | "
                    f"improvements={improvement_count} | "
                    f"no-improve={no_improve} | "
                    f"elapsed={elapsed:.1f}s | "
                    f"ETA≈{remaining:.1f}s",
                    flush=True,
                )

        return current, current_score

    def optimize(self) -> None:
        for restart in range(self.restarts):
            rng = random.Random(self.seed + restart)

            initial = self.greedy_initial_split(rng)

            if initial is None:
                print(
                    f"Restart {restart + 1}/{self.restarts}: "
                    "initialization failed"
                )
                continue

            initial_score = self.score(initial)

            print(
                f"\nRestart {restart + 1}/{self.restarts} started | "
                f"initial score={initial_score:.10f}",
                flush=True,
            )

            optimized, final_score = self.fast_local_search(
                initial,
                rng,
                restart_index=restart + 1,
            )

            print(
                f"Restart {restart + 1}/{self.restarts} completed: "
                f"{initial_score:.10f} -> {final_score:.10f}",
                flush=True,
            )

            if final_score < self.best_score:
                self.best_score = final_score
                self.best_test_set = optimized

        if self.best_test_set is None:
            raise RuntimeError("未找到满足约束的划分方案。")

        print("\n===== Optimization completed =====")
        print(f"Best score: {self.best_score:.10f}")

    def save_results(self) -> None:
        if self.best_test_set is None:
            raise RuntimeError("请先运行 optimize()。")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        test_set = self.best_test_set
        train_set = set(self.all_images) - test_set

        pd.Series(
            sorted(train_set)
        ).to_csv(
            self.output_dir / "train_images.csv",
            index=False,
            header=False,
        )

        pd.Series(
            sorted(test_set)
        ).to_csv(
            self.output_dir / "test_images.csv",
            index=False,
            header=False,
        )

        train_df = self.df[
            self.df[self.filename_column].isin(train_set)
        ]

        test_df = self.df[
            self.df[self.filename_column].isin(test_set)
        ]

        train_image_counts = (
            train_df[
                [self.filename_column, self.class_column]
            ]
            .drop_duplicates()
            [self.class_column]
            .value_counts()
            .reindex(self.all_classes, fill_value=0)
        )

        test_image_counts = (
            test_df[
                [self.filename_column, self.class_column]
            ]
            .drop_duplicates()
            [self.class_column]
            .value_counts()
            .reindex(self.all_classes, fill_value=0)
        )

        train_instance_counts = (
            train_df[self.class_column]
            .value_counts()
            .reindex(self.all_classes, fill_value=0)
        )

        test_instance_counts = (
            test_df[self.class_column]
            .value_counts()
            .reindex(self.all_classes, fill_value=0)
        )

        report = pd.DataFrame(index=self.all_classes)
        report.index.name = self.class_column

        report["total_images"] = [
            self.class_total_images[c]
            for c in self.all_classes
        ]
        report["train_images"] = train_image_counts
        report["test_images"] = test_image_counts
        report["test_image_ratio"] = (
            report["test_images"] / report["total_images"]
        )

        report["total_instances"] = [
            self.class_total_instances[c]
            for c in self.all_classes
        ]
        report["train_instances"] = train_instance_counts
        report["test_instances"] = test_instance_counts
        report["test_instance_ratio"] = (
            report["test_instances"]
            / report["total_instances"]
        )

        report["both_sides_covered"] = (
            (report["train_images"] > 0)
            & (report["test_images"] > 0)
        )

        report.to_csv(
            self.output_dir / "class_split_report.csv",
            encoding="utf-8-sig",
        )

        summary = pd.DataFrame(
            [
                ["total_images", len(self.all_images)],
                ["train_images", len(train_set)],
                ["test_images", len(test_set)],
                [
                    "test_image_ratio",
                    len(test_set) / len(self.all_images),
                ],
                ["total_instances", len(self.df)],
                ["train_instances", len(train_df)],
                ["test_instances", len(test_df)],
                [
                    "test_instance_ratio",
                    len(test_df) / len(self.df),
                ],
                [
                    "train_classes",
                    train_df[self.class_column].nunique(),
                ],
                [
                    "test_classes",
                    test_df[self.class_column].nunique(),
                ],
                ["best_score", self.best_score],
            ],
            columns=["metric", "value"],
        )

        summary.to_csv(
            self.output_dir / "split_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print("\n===== Saved files =====")
        print((self.output_dir / "train_images.csv").resolve())
        print((self.output_dir / "test_images.csv").resolve())
        print((self.output_dir / "class_split_report.csv").resolve())
        print((self.output_dir / "split_summary.csv").resolve())

        print("\n===== Validation =====")
        print(f"Train images: {len(train_set)}")
        print(f"Test images: {len(test_set)}")
        print(
            f"Train classes: "
            f"{train_df[self.class_column].nunique()}"
        )
        print(
            f"Test classes: "
            f"{test_df[self.class_column].nunique()}"
        )

        missing_train = report[report["train_images"] == 0]
        missing_test = report[report["test_images"] == 0]

        print(
            "Classes missing from train:",
            missing_train.index.tolist(),
        )
        print(
            "Classes missing from test:",
            missing_test.index.tolist(),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gt",
        type=Path,
        default=Path("GT.csv"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("optimized_split_80_20_fast"),
    )

    parser.add_argument(
        "--filename-column",
        type=str,
        default="filename",
    )

    parser.add_argument(
        "--class-column",
        type=str,
        default="Class ID",
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--restarts",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--swap-rounds",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--candidate-pool",
        type=int,
        default=20,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    optimizer = FastDatasetSplitOptimizer(
        gt_csv=args.gt,
        output_dir=args.output,
        filename_column=args.filename_column,
        class_column=args.class_column,
        train_ratio=args.train_ratio,
        seed=args.seed,
        restarts=args.restarts,
        swap_rounds=args.swap_rounds,
        candidate_pool=args.candidate_pool,
    )

    optimizer.load_data()
    optimizer.optimize()
    optimizer.save_results()


if __name__ == "__main__":
    main()
