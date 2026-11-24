"""Two-tower neural recommender implemented with PyTorch."""

from collections.abc import Iterable
from pathlib import Path
from itertools import islice

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class TwoTowerModel(nn.Module):
    """Learn separate user and item representations whose dot product is a score."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float,
        user_features: np.ndarray | None = None,
        item_features: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        user_feature_dim = 0 if user_features is None else user_features.shape[1]
        item_feature_dim = 0 if item_features is None else item_features.shape[1]
        self.register_buffer(
            "user_features",
            torch.empty((n_users, 0), dtype=torch.float32)
            if user_features is None
            else torch.tensor(user_features, dtype=torch.float32),
        )
        self.register_buffer(
            "item_features",
            torch.empty((n_items, 0), dtype=torch.float32)
            if item_features is None
            else torch.tensor(item_features, dtype=torch.float32),
        )
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.user_tower = nn.Sequential(
            nn.Linear(embedding_dim + user_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.item_tower = nn.Sequential(
            nn.Linear(embedding_dim + item_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.user_embedding.weight, std=0.05)
        nn.init.normal_(self.item_embedding.weight, std=0.05)
        for module in (*self.user_tower, *self.item_tower):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def encode_users(self, user_indices: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat(
            (self.user_embedding(user_indices), self.user_features[user_indices]),
            dim=-1,
        )
        return self.user_tower(inputs)

    def encode_items(self, item_indices: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat(
            (self.item_embedding(item_indices), self.item_features[item_indices]),
            dim=-1,
        )
        return self.item_tower(inputs)

    def forward(
        self,
        user_indices: torch.Tensor,
        item_indices: torch.Tensor,
    ) -> torch.Tensor:
        user_vectors = self.encode_users(user_indices)
        item_vectors = self.encode_items(item_indices)
        return (user_vectors * item_vectors).sum(dim=-1)


class TwoTowerRecommender:
    """Train and use a two-tower recommender on MovieLens ratings."""

    def __init__(
        self,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        epochs: int = 5,
        batch_size: int = 4096,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        negative_ratio: int = 1,
        loss: str = "bce",
        positive_threshold: float = 4.0,
        dropout: float = 0.1,
        seed: int = 42,
        device: str | None = None,
    ) -> None:
        if embedding_dim < 1 or hidden_dim < 1:
            raise ValueError("embedding_dim and hidden_dim must be positive.")
        if epochs < 1 or batch_size < 1 or negative_ratio < 1:
            raise ValueError("epochs, batch_size, and negative_ratio must be positive.")
        if learning_rate <= 0 or weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative.")
        if loss not in {"bce", "bpr"}:
            raise ValueError("loss must be 'bce' or 'bpr'.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in the range [0, 1).")

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.negative_ratio = negative_ratio
        self.loss = loss
        self.positive_threshold = positive_threshold
        self.dropout = dropout
        self.seed = seed
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._fitted = False

    def fit(
        self,
        ratings: pd.DataFrame,
        catalogue: Iterable[int] | None = None,
        user_features: pd.DataFrame | None = None,
        item_features: pd.DataFrame | None = None,
    ) -> "TwoTowerRecommender":
        """Fit on positive interactions and sampled unobserved items."""
        required_columns = {"UserID", "MovieID", "Rating"}
        missing = required_columns.difference(ratings.columns)
        if missing:
            raise ValueError(f"ratings is missing columns: {sorted(missing)}")

        positives = ratings[ratings["Rating"] >= self.positive_threshold]
        if positives.empty:
            raise ValueError("Cannot fit two-tower model with no positive interactions.")

        catalogue_ids = pd.Index(
            catalogue if catalogue is not None else ratings["MovieID"].unique(),
            name="MovieID",
        ).drop_duplicates()
        train_item_ids = pd.Index(positives["MovieID"].unique())
        catalogue_ids = catalogue_ids.append(train_item_ids).drop_duplicates()
        user_ids = pd.Index(positives["UserID"].unique(), name="UserID")
        self.user_index_ = {int(value): index for index, value in enumerate(user_ids)}
        self.item_index_ = {int(value): index for index, value in enumerate(catalogue_ids)}
        self.user_ids_ = user_ids.to_numpy()
        self.item_ids_ = catalogue_ids.to_numpy()
        self.train_ratings_ = ratings.copy()
        user_feature_values = self._features_for_ids(user_ids, user_features)
        item_feature_values = self._features_for_ids(catalogue_ids, item_features)

        n_items = len(catalogue_ids)
        positive_users = positives["UserID"].map(self.user_index_).to_numpy(dtype=np.int64)
        positive_items = positives["MovieID"].map(self.item_index_).to_numpy(dtype=np.int64)
        known_ratings = ratings[
            ratings["UserID"].isin(self.user_index_)
            & ratings["MovieID"].isin(self.item_index_)
        ]
        # Every rated (user, item) pair encoded as one integer for vectorised lookups.
        seen_keys = np.unique(
            known_ratings["UserID"].map(self.user_index_).to_numpy(dtype=np.int64) * n_items
            + known_ratings["MovieID"].map(self.item_index_).to_numpy(dtype=np.int64)
        )
        # Users who have rated the whole catalogue have no negatives to sample.
        seen_counts = np.bincount(seen_keys // n_items, minlength=len(user_ids))
        has_negatives = seen_counts[positive_users] < n_items
        positive_users = positive_users[has_negatives]
        positive_items = positive_items[has_negatives]

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.model_ = TwoTowerModel(
            n_users=len(user_ids),
            n_items=n_items,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            user_features=user_feature_values,
            item_features=item_feature_values,
        ).to(self.device)
        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loss_function = nn.BCEWithLogitsLoss()
        rng = np.random.default_rng(self.seed)
        self.history_: list[float] = []

        for epoch in range(1, self.epochs + 1):
            user_indices, positive_examples, negative_items = self._make_training_examples(
                positive_users, positive_items, seen_keys, n_items, rng
            )
            dataset = TensorDataset(
                torch.tensor(user_indices, dtype=torch.long),
                torch.tensor(positive_examples, dtype=torch.long),
                torch.tensor(negative_items, dtype=torch.long),
            )
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            self.model_.train()
            total_loss = 0.0
            total_examples = 0
            for batch_users, batch_positive_items, batch_negative_items in loader:
                batch_users = batch_users.to(self.device)
                batch_positive_items = batch_positive_items.to(self.device)
                batch_negative_items = batch_negative_items.to(self.device)
                optimizer.zero_grad()
                positive_scores = self.model_(batch_users, batch_positive_items)
                negative_scores = self.model_(batch_users, batch_negative_items)
                if self.loss == "bpr":
                    batch_loss = -torch.nn.functional.logsigmoid(
                        positive_scores - negative_scores
                    ).mean()
                else:
                    scores = torch.cat((positive_scores, negative_scores))
                    labels = torch.cat(
                        (
                            torch.ones_like(positive_scores),
                            torch.zeros_like(negative_scores),
                        )
                    )
                    batch_loss = loss_function(scores, labels)
                batch_loss.backward()
                optimizer.step()
                total_loss += float(batch_loss.item()) * len(batch_users)
                total_examples += len(batch_users)
            epoch_loss = total_loss / total_examples
            self.history_.append(epoch_loss)
            print(f"Two-tower epoch {epoch}/{self.epochs}: loss={epoch_loss:.4f}")

        self._fitted = True
        return self

    @staticmethod
    def _features_for_ids(
        ids: pd.Index,
        features: pd.DataFrame | None,
    ) -> np.ndarray | None:
        if features is None:
            return None
        if features.index.has_duplicates:
            raise ValueError("Feature indices must contain unique IDs.")
        values = features.reindex(ids).fillna(0.0)
        if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in values.dtypes):
            raise ValueError("Feature values must be numeric.")
        return values.to_numpy(dtype=np.float32)

    def recommend(self, user_id: int, n: int = 10) -> list[int]:
        """Return up to n highest-scoring unseen items for a known user."""
        self._check_fitted()
        if n < 1:
            raise ValueError("n must be positive.")
        user_index = self.user_index_.get(int(user_id))
        if user_index is None:
            return []

        item_indices = torch.arange(len(self.item_ids_), device=self.device)
        user_indices = torch.full_like(item_indices, user_index)
        self.model_.eval()
        with torch.no_grad():
            scores = self.model_(user_indices, item_indices).cpu().numpy()

        seen_items = set(
            self.train_ratings_.loc[
                self.train_ratings_["UserID"] == user_id, "MovieID"
            ].astype(int)
        )
        ranked_indices = np.argsort(-scores)
        return [
            int(self.item_ids_[index])
            for index in ranked_indices
            if int(self.item_ids_[index]) not in seen_items
        ][:n]

    def recommend_with_scores(self, user_id: int, n: int = 10) -> list[int]:
        """Return up to n highest-scoring unseen items and their scores for a known user."""
        self._check_fitted()
        if n < 1:
            raise ValueError("n must be positive.")
        user_index = self.user_index_.get(int(user_id))
        if user_index is None:
            return []

        item_indices = torch.arange(len(self.item_ids_), device=self.device)
        user_indices = torch.full_like(item_indices, user_index)
        self.model_.eval()
        with torch.no_grad():
            scores = self.model_(user_indices, item_indices).cpu().numpy()

        seen_items = set(
            self.train_ratings_.loc[
                self.train_ratings_["UserID"] == user_id, "MovieID"
            ].astype(int)
        )
        ranked_indices = np.argsort(-scores)
        all_recs = {
            int(self.item_ids_[index]) : scores[index]
            for index in ranked_indices
            if int(self.item_ids_[index]) not in seen_items
        }
        return dict(islice(all_recs.items(), n)) 

    def recommend_all(
        self, user_ids: Iterable[int], n: int = 10
    ) -> dict[int, list[int]]:
        """Return recommendations for each requested user."""
        return {int(user_id): self.recommend(int(user_id), n=n) for user_id in user_ids}

    def recommend_all_with_scores(
        self, user_ids: Iterable[int], n: int = 10
    ) -> dict[int, list[int]]:
        """Return recommendations for each requested user as dict {item_idx : score}."""
        return {int(user_id): self.recommend_with_scores(int(user_id), n=n) for user_id in user_ids}

    def save(self, path: str | Path) -> None:
        """Save the fitted model and recommender state to a local checkpoint."""
        self._check_fitted()
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model_.state_dict(),
                "user_index": self.user_index_,
                "item_index": self.item_index_,
                "user_ids": self.user_ids_,
                "item_ids": self.item_ids_,
                "train_ratings": self.train_ratings_,
                "history": self.history_,
                "config": {
                    "embedding_dim": self.embedding_dim,
                    "hidden_dim": self.hidden_dim,
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                    "learning_rate": self.learning_rate,
                    "weight_decay": self.weight_decay,
                    "negative_ratio": self.negative_ratio,
                    "loss": self.loss,
                    "positive_threshold": self.positive_threshold,
                    "dropout": self.dropout,
                    "seed": self.seed,
                },
            },
            checkpoint_path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "TwoTowerRecommender":
        """Restore a recommender written by save()."""
        # The checkpoint holds pandas and numpy objects, so weights-only loading cannot read it.
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
        recommender = cls(**checkpoint["config"], device=device)
        state = checkpoint["model_state_dict"]
        # Feature buffers live in the state dict; zero-width buffers mean no side features.
        user_features = state["user_features"].numpy()
        item_features = state["item_features"].numpy()
        recommender.model_ = TwoTowerModel(
            n_users=len(checkpoint["user_ids"]),
            n_items=len(checkpoint["item_ids"]),
            embedding_dim=recommender.embedding_dim,
            hidden_dim=recommender.hidden_dim,
            dropout=recommender.dropout,
            user_features=user_features if user_features.shape[1] else None,
            item_features=item_features if item_features.shape[1] else None,
        ).to(recommender.device)
        recommender.model_.load_state_dict(state)
        recommender.user_index_ = checkpoint["user_index"]
        recommender.item_index_ = checkpoint["item_index"]
        recommender.user_ids_ = checkpoint["user_ids"]
        recommender.item_ids_ = checkpoint["item_ids"]
        recommender.train_ratings_ = checkpoint["train_ratings"]
        recommender.history_ = checkpoint["history"]
        recommender._fitted = True
        return recommender

    def _make_training_examples(
        self,
        positive_users: np.ndarray,
        positive_items: np.ndarray,
        seen_keys: np.ndarray,
        n_items: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pair each positive with negative_ratio items drawn uniformly from the user's unseen items."""
        user_indices = np.repeat(positive_users, self.negative_ratio)
        positive_examples = np.repeat(positive_items, self.negative_ratio)
        negative_items = rng.integers(0, n_items, size=len(user_indices))
        # Rejection sampling: redraw only the negatives the user has already rated.
        collisions = np.isin(user_indices * n_items + negative_items, seen_keys)
        while collisions.any():
            negative_items[collisions] = rng.integers(0, n_items, size=int(collisions.sum()))
            collisions[collisions] = np.isin(
                user_indices[collisions] * n_items + negative_items[collisions], seen_keys
            )
        return user_indices, positive_examples, negative_items

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Fit the two-tower model before requesting recommendations.")
