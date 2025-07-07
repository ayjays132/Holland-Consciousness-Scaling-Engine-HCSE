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
