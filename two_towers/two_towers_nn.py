"""Two-tower neural recommender implemented with PyTorch."""

from collections.abc import Iterable
from pathlib import Path

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
    ) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.user_tower = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.item_tower = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
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
        return self.user_tower(self.user_embedding(user_indices))

    def encode_items(self, item_indices: torch.Tensor) -> torch.Tensor:
        return self.item_tower(self.item_embedding(item_indices))

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
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in the range [0, 1).")

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.negative_ratio = negative_ratio
        self.positive_threshold = positive_threshold
        self.dropout = dropout
        self.seed = seed
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._fitted = False

    def fit(
        self,
        ratings: pd.DataFrame,
        catalogue: Iterable[int] | None = None,
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

        positive_pairs = [
            (self.user_index_[int(user_id)], self.item_index_[int(item_id)])
            for user_id, item_id in positives[["UserID", "MovieID"]].itertuples(
                index=False, name=None
            )
        ]
        known_ratings = ratings[ratings["UserID"].isin(self.user_index_)]
        seen_by_user = {
            user_index: {
                self.item_index_[int(item_id)]
                for item_id in user_ratings["MovieID"]
                if int(item_id) in self.item_index_
            }
            for user_index, user_ratings in known_ratings.groupby(
                known_ratings["UserID"].map(self.user_index_)
            )
        }

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.model_ = TwoTowerModel(
            n_users=len(user_ids),
            n_items=len(catalogue_ids),
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
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
            user_indices, item_indices, labels = self._make_training_examples(
                positive_pairs, seen_by_user, len(catalogue_ids), rng
            )
            dataset = TensorDataset(
                torch.tensor(user_indices, dtype=torch.long),
                torch.tensor(item_indices, dtype=torch.long),
                torch.tensor(labels, dtype=torch.float32),
            )
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            self.model_.train()
            total_loss = 0.0
            total_examples = 0
            for batch_users, batch_items, batch_labels in loader:
                batch_users = batch_users.to(self.device)
                batch_items = batch_items.to(self.device)
                batch_labels = batch_labels.to(self.device)
                optimizer.zero_grad()
                loss = loss_function(
                    self.model_(batch_users, batch_items), batch_labels
                )
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item()) * len(batch_labels)
                total_examples += len(batch_labels)
            epoch_loss = total_loss / total_examples
            self.history_.append(epoch_loss)
            print(f"Two-tower epoch {epoch}/{self.epochs}: loss={epoch_loss:.4f}")

        self._fitted = True
        return self

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

    def recommend_all(
        self, user_ids: Iterable[int], n: int = 10
    ) -> dict[int, list[int]]:
        """Return recommendations for each requested user."""
        return {int(user_id): self.recommend(int(user_id), n=n) for user_id in user_ids}

    def _make_training_examples(
        self,
        positive_pairs: list[tuple[int, int]],
        seen_by_user: dict[int, set[int]],
        n_items: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        user_indices: list[int] = []
        item_indices: list[int] = []
        labels: list[float] = []
        all_items = np.arange(n_items)
        for user_index, item_index in positive_pairs:
            user_indices.append(user_index)
            item_indices.append(item_index)
            labels.append(1.0)
            available = np.setdiff1d(all_items, seen_by_user[user_index], assume_unique=True)
            if len(available) == 0:
                continue
            negative_items = rng.choice(
                available,
                size=self.negative_ratio,
                replace=len(available) < self.negative_ratio,
            )
            user_indices.extend([user_index] * self.negative_ratio)
            item_indices.extend(int(value) for value in negative_items)
            labels.extend([0.0] * self.negative_ratio)
        return (
            np.asarray(user_indices, dtype=np.int64),
            np.asarray(item_indices, dtype=np.int64),
            np.asarray(labels, dtype=np.float32),
        )

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Fit the two-tower model before requesting recommendations.")
