import torch
import torch.nn as nn
import logging
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, random_split
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_NAME = "HuggingFaceTB/SmolLM-135M"
MAX_LENGTH = 2048
BATCH_SIZE = 12
LEARNING_RATE = 2e-4
MAX_STEPS = 3000
GRADIENT_ACCUMULATION_STEPS = 2
NUM_WARMUP_STEPS = 30
OUTPUT_DIR = "./longcustom_finetuned_results"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def data_collator(features):
    input_ids = torch.stack([f[0] for f in features])
    attention_mask = torch.stack([f[1] for f in features])
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': input_ids.clone(),
    }
class GrokAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2,
                 alpha_init=0.98, lamb=2.0, gamma=0.1, grokking_signal_fns=None,
                 grokking_signal_decay_rate=0.1, gradient_clipping=1.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        alpha_init=alpha_init, lamb=lamb, gamma=gamma,
                        grokking_signal_fns=grokking_signal_fns,
                        grokking_signal_decay_rate=grokking_signal_decay_rate,
                        gradient_clipping=gradient_clipping)
        super(GrokAdamW, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            grokking_signal = self._compute_grokking_signal(group)
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                grad = p.grad

                if group['gradient_clipping'] > 0:
                    grad = torch.clamp(grad, -group['gradient_clipping'], group['gradient_clipping'])

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['grok_ema'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq, grok_ema = state['exp_avg'], state['exp_avg_sq'], state['grok_ema']
                beta1, beta2 = group['betas']

                state['step'] += 1
                
                # Layer-wise momentum decay
                layer_beta1 = beta1 * (1 - group['gamma'])**i

                # Grokfast component
                alpha = group['alpha_init'] * torch.exp(torch.tensor(-group['grokking_signal_decay_rate'] * grokking_signal))
                grok_ema.mul_(alpha).add_(grad, alpha=1 - alpha)
                grok_grad = grad + group['lamb'] * grok_ema

                # AdamW update with Grokfast-amplified gradient
                exp_avg.mul_(layer_beta1).add_(grok_grad, alpha=1 - layer_beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grok_grad, grok_grad, value=1 - beta2)

                denom = exp_avg_sq.sqrt().add_(group['eps'])
                step_size = group['lr']

                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

    def _compute_grokking_signal(self, group):
        if group['grokking_signal_fns'] is None:
            return 0.0

        signals = []
        for fn in group['grokking_signal_fns']:
            try:
                signal = fn()
                if signal is not None:
                    signals.append(signal)
            except Exception as e:
                logger.warning(f"Error in grokking_signal_fn: {e}. Ignoring this function.")

        if not signals:
            return 0.0

        return sum(signals) / len(signals)

def load_custom_dataset():
    logger.info("🔍 Loading Reddit jokes dataset")
    try:
        # Load the dataset
        data = load_dataset("SocialGrep/one-million-reddit-jokes", split='train')
        
        # Filter and format the data
        texts = []
        for item in tqdm(data, desc="Formatting jokes"):
            title = item['title'] or ""
            selftext = item['selftext'] or ""
            if len(title + selftext) > 0:
                formatted_text = f"Human: Tell me a joke\n\nAssistant: Here's a joke from Reddit:\n\n"
                if title:
                    formatted_text += f"{title}\n\n"
                if selftext:
                    formatted_text += f"{selftext}\n\n"
                texts.append(formatted_text.strip())
        
        # Create a new dataset
        return data.select(range(len(texts))).map(
            lambda example, idx: {"text": texts[idx]},
            with_indices=True
        )
    except Exception as e:
        logger.error(f"❌ Failed to load Reddit jokes dataset: {str(e)}")
        return None

def format_capybara_prompts(examples):
    texts = []
    for conversation in examples['conversation']:
        formatted_text = ""
        for turn in conversation:
            if 'input' in turn:
                formatted_text += f"Human: {turn['input']}\n\n"
            if 'output' in turn:
                formatted_text += f"Assistant: {turn['output']}\n\n"
        texts.append(formatted_text.strip())
    return {"text": texts}

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grokking_signal = 0.0

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)

        with autocast(dtype=torch.bfloat16):
            loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()

        self.grokking_signal = loss.item()

        return loss.detach()

def grokking_signal_fn():
    return trainer.grokking_signal

def main():
    logger.info(f"🚀 Initializing {MODEL_NAME} finetuning with GrokAdamW")
    
    try:
        config = AutoConfig.from_pretrained(MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
        
    except Exception as e:
        logger.error(f"❌ Failed to load model or tokenizer: {str(e)}")
        return

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    logger.info("📚 Loading datasets")
    custom_dataset = load_custom_dataset()
    if custom_dataset is None:
        logger.error("❌ Failed to load custom dataset. Aborting.")
        return

    try:
        capybara_dataset = load_dataset("LDJnr/Capybara", split="train")
        capybara_dataset = capybara_dataset.map(format_capybara_prompts, batched=True, remove_columns=capybara_dataset.column_names)
    except Exception as e:
        logger.error(f"❌ Failed to load Capybara dataset: {str(e)}")
        capybara_dataset = Dataset.from_dict({"text": []})

    logger.info(f"📊 Custom dataset size: {len(custom_dataset)}")
    logger.info(f"📊 Capybara dataset size: {len(capybara_dataset)}")

    combined_dataset = concatenate_datasets([custom_dataset, capybara_dataset])
    combined_dataset = combined_dataset.shuffle(seed=42)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)

    logger.info("🔢 Tokenizing dataset")
    tokenized_dataset = combined_dataset.map(tokenize_function, batched=True, remove_columns=combined_dataset.column_names)

    logger.info("🏋️ Setting up the training arguments")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        bf16=True,
        logging_steps=10,
        save_steps=300,
        save_total_limit=10,
        dataloader_num_workers=4,
        warmup_steps=NUM_WARMUP_STEPS,
        gradient_checkpointing=True,
        evaluation_strategy="steps",
        eval_steps=300,
        max_steps=MAX_STEPS,
        fp16=False,  # We're using bf16, so disable fp16
        optim="adamw_hf",  # Using the huggingface one allows us to use custom optimizers
        lr_scheduler_type="cosine",  # Cosine learning rate decay
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    optimizer = GrokAdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        alpha_init=0.98,
        lamb=2.0,
        gamma=0.1,
        grokking_signal_fns=[grokking_signal_fn],
        grokking_signal_decay_rate=0.1,
        gradient_clipping=1.0
    )

    logger.info("🏃‍♂️ Initializing Trainer with GrokAdamW")
    global trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset.select(range(min(1000, len(tokenized_dataset)))),
        data_collator=data_collator,
        optimizers=(optimizer, None), # This line tells it to use GrokAdamW
    )

    logger.info("🔥 Starting the training with GrokAdamW")
    try:
        trainer.train()
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        return

    logger.info("💾 Saving the model")
    try:
        trainer.save_model(OUTPUT_DIR)
    except Exception as e:
        logger.error(f"❌ Failed to save model: {str(e)}")

    logger.info("🎉 Finetuning with GrokAdamW completed!")

if __name__ == "__main__":
    main()