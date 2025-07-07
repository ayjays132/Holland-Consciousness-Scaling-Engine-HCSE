"""Example usage of HCSE with a HuggingFace model."""

from hcse.core import HCSEMixin
from hcse.pipeline import HfTrainerWithHCSE
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments


class ModelWithHCSE(HCSEMixin, AutoModelForCausalLM):
    pass


def main():
    model_name = "sshleifer/tiny-gpt2"
    model = ModelWithHCSE.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=32)

    dataset = [{"text": "Hello world"}]
    dataset = [tokenize(item) for item in dataset]

    args = TrainingArguments(output_dir="./hcse-example", per_device_train_batch_size=1, num_train_epochs=1)
    trainer = HfTrainerWithHCSE(model=model, args=args, train_dataset=dataset, hcse_params={"layer": -1})
    trainer.train()


if __name__ == "__main__":
    main()
