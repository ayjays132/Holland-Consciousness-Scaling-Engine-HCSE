import torch
from transformers import TrainingArguments, TrainerCallback

from hcse.core import HCSEMixin
from hcse.pipeline import HfTrainerWithHCSE


class BaseModel(torch.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor, output_hidden_states: bool = False):
        hidden = self.linear(input_ids)
        loss = torch.nn.functional.mse_loss(hidden, labels)
        outputs = {
            "loss": loss,
            "hidden_states": (hidden,),
        }
        return type("Output", (), outputs)


class DummyModel(HCSEMixin, BaseModel):
    def __init__(self, hidden_size: int) -> None:
        HCSEMixin.__init__(self, hidden_size=hidden_size)


class DummyDataset(torch.utils.data.Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, idx: int):
        x = torch.randn(4)
        return {"input_ids": x, "labels": x}


def test_trainer_integration():
    model = DummyModel(4)
    args = TrainingArguments(output_dir="/tmp/hcse-tests", per_device_train_batch_size=1, num_train_epochs=1)
    trainer = HfTrainerWithHCSE(model=model, args=args, train_dataset=DummyDataset(), hcse_params={"layer": 0})
    trainer.add_callback(TrainerCallback())
    result = trainer.train(resume_from_checkpoint=None)
    assert result.training_loss is not None

def test_bonus_modifies_loss():
    model = DummyModel(4)
    args = TrainingArguments(output_dir="/tmp/hcse-tests", per_device_train_batch_size=1)
    ds = DummyDataset()
    trainer_no_bonus = HfTrainerWithHCSE(model=model, args=args, train_dataset=ds, hcse_params={"layer": 0, "lambda_c": 0.0})
    trainer_bonus = HfTrainerWithHCSE(model=model, args=args, train_dataset=ds, hcse_params={"layer": 0, "lambda_c": 1.0})
    sample = [ds[0], ds[1]]
    batch = {k: torch.stack([s[k] for s in sample]) for k in sample[0]}
    loss_no_bonus = trainer_no_bonus.compute_loss(model, batch)
    loss_bonus = trainer_bonus.compute_loss(model, batch)
    assert loss_bonus < loss_no_bonus

def test_compute_loss_uses_bonus():
    model = DummyModel(4)
    args = TrainingArguments(
        output_dir="/tmp/hcse-tests",
        per_device_train_batch_size=1,
        num_train_epochs=1,
    )
    trainer = HfTrainerWithHCSE(
        model=model,
        args=args,
        train_dataset=DummyDataset(),
        hcse_params={"layer": 0, "lambda_c": 1.0},
    )

    inputs = {"input_ids": torch.randn(2, 4), "labels": torch.randn(2, 4)}
    baseline_loss = model.forward(**inputs, output_hidden_states=True).loss
    expected_loss = model.forward_with_hcse(
        **inputs, hcse_params={"layer": 0, "lambda_c": 1.0}
    ).loss
    trainer_loss = trainer.compute_loss(model, inputs)

    assert torch.allclose(trainer_loss, expected_loss)
    assert trainer_loss < baseline_loss
