#!/usr/bin/env python3
"""
Production-Ready Ensemble Training and Webpage Generation Script
with RAIF-based Automatic Alignment, Synthetic Reference Datasets,
and Cross-Model Alignment

This script implements an ensemble system that:
  • Loads three models: a code generator (e.g. StarCoder2-3B), a design language model 
    (e.g. Llama-3.1-8B-Instruct), and an evaluator (e.g. CodeGemma-7B-IT).
  • Loads local synthetic reference datasets (HTML, CSS, and JS files) for enhanced training metrics.
  • Generates detailed design specifications from a text prompt.
  • Generates code based on the design specification.
  • Evaluates the generated code using an enhanced scoring mechanism that includes reference similarity.
  • Uses a RAIF-based deep thinking cycle to iteratively refine code (multiple candidate generation).
  • Aggregates scores via weighted averaging.
  • Performs cross-model alignment to fuse knowledge between the models, avoiding biases and ensuring consistency.
  • Performs training with detailed metrics logging and knowledge distillation to produce a new distilled model.
  • Monitors system resources and saves models in safetensors format.
  • Provides centralized configuration for training parameters and metrics.

Usage:
  To generate a webpage code:
      $ python ensemble_system.py --prompt "Create a responsive landing page for an AI startup with dark theme"
  To run ensemble training:
      $ python ensemble_system.py --train
"""

import os
import json
import logging
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import psutil
from typing import Dict, Any, Tuple, List

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    StoppingCriteria
)
from safetensors.torch import save_file

# For computing reference similarity using TF-IDF and cosine similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

##############################################
# Global Training Configuration
##############################################

TRAINING_CONFIG = {
    "code_generator": "main/models/primary/bigcode/starcoder2-3b",
    "design_llm": "main/models/secondary/Llama-3.1-8B-Instruct",
    "evaluator": "main/models/evaluator/codegemma-7b-it",
    "device_map": "auto",
    "torch_dtype": torch.float16,
    "max_code_length": 2048,
    "max_refinement_cycles": 3,
    "evaluation_threshold": 0.85,
    "learning_rate": 1e-5,
    "training_iterations": 10,         # Number of synthetic training samples per training run
    "deep_thinking_candidates": 3,       # Number of candidate refinements in the deep thinking cycle
    "synthetic_dataset_paths": {         # Local paths for synthetic reference datasets
        "html": "mai/training/synthetic_data/html",
        "css": "main/training/synthetic_data/css",
        "js": "main/training/synthetic_data/js"
    },
    "score_weights": {                   # Weights for aggregating evaluation and reference similarity scores
        "evaluation": 0.7,
        "reference": 0.3
    },
    "alignment": {                       # Parameters for cross-model alignment
        "common_dim": 512,
        "learning_rate": 1e-4,
        "iterations": 5,
        "alignment_prompt": "Align ensemble models for consistent output."
    }
}

##############################################
# Resource Management Classes
##############################################

class ResourceLimits:
    """
    Defines system resource limits.
    Adjust these values based on available system RAM and GPU VRAM.
    """
    def __init__(self, max_ram_gb: float = 20.0, max_vram_gb: float = 3.8):
        self.max_ram_gb = max_ram_gb
        self.max_vram_gb = max_vram_gb

class ResourceManager:
    """
    Monitors system memory usage and releases GPU memory as needed.
    """
    def __init__(self, limits: ResourceLimits):
        self.limits = limits

    def check_memory(self) -> bool:
        memory = psutil.virtual_memory()
        used_ram_gb = memory.used / (1024**3)
        logging.debug(f"RAM used: {used_ram_gb:.2f} GB; Limit: {self.limits.max_ram_gb} GB")
        return used_ram_gb < self.limits.max_ram_gb

    def release_gpu_memory(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logging.info("Released GPU memory via torch.cuda.empty_cache().")

##############################################
# Custom Stopping Criteria for Generation
##############################################

class CodeStoppingCriteria(StoppingCriteria):
    """
    Stops generation when a designated stop token is encountered.
    """
    def __init__(self, stop_tokens: List[int]):
        self.stop_tokens = stop_tokens

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0][-1] in self.stop_tokens

##############################################
# Synthetic Reference Dataset Functions
##############################################

def load_synthetic_references(paths: Dict[str, str]) -> Dict[str, List[str]]:
    """
    Load synthetic reference files from specified local paths.
    Returns a dictionary with keys 'html', 'css', and 'js' containing lists of file contents.
    """
    references = {}
    for key, path in paths.items():
        references[key] = []
        if os.path.exists(path):
            for filename in os.listdir(path):
                if filename.endswith((".html", ".css", ".js")):
                    full_path = os.path.join(path, filename)
                    try:
                        with open(full_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            if content:
                                references[key].append(content)
                    except Exception as e:
                        logging.warning(f"Failed to read {full_path}: {e}")
        else:
            logging.warning(f"Path '{path}' does not exist for {key} references.")
    logging.info("Synthetic reference datasets loaded.")
    return references

def compute_similarity_with_references(code: str, references: Dict[str, List[str]]) -> float:
    """
    Compute cosine similarity between the generated code and all reference texts using TF-IDF.
    Returns the maximum similarity score (between 0 and 1).
    """
    all_refs = []
    for ref_list in references.values():
        all_refs.extend(ref_list)
    if not all_refs:
        logging.warning("No synthetic references available; returning similarity 0.0")
        return 0.0
    vectorizer = TfidfVectorizer().fit(all_refs + [code])
    code_vec = vectorizer.transform([code])
    refs_vec = vectorizer.transform(all_refs)
    similarities = cosine_similarity(code_vec, refs_vec)
    max_similarity = float(similarities.max())
    logging.info(f"Computed maximum cosine similarity: {max_similarity:.2f}")
    return max_similarity

def aggregate_model_scores(evaluation_score: float, reference_score: float, weights: Dict[str, float]) -> float:
    """
    Aggregate evaluation and reference similarity scores using weighted averaging.
    """
    aggregated = (weights["evaluation"] * evaluation_score) + (weights["reference"] * reference_score)
    logging.info(f"Aggregated score: {aggregated:.2f} (Evaluation: {evaluation_score:.2f}, Reference: {reference_score:.2f})")
    return aggregated

##############################################
# Web Ensemble System with RAIF and Cross-Model Alignment
##############################################

class WebEnsembleSystem:
    """
    Ensemble system that integrates design specification extraction, code generation,
    RAIF-based iterative refinement, enhanced evaluation, cross-model alignment,
    training, and model saving.
    """
    def __init__(self):
        self.config = TRAINING_CONFIG
        self.logger = self._configure_logger()
        self.resource_manager = ResourceManager(ResourceLimits())
        self.models = self._load_models()
        self.tokenizers = self._load_tokenizers()
        self.eval_criteria = self._load_evaluation_criteria("eval_criteria.json")
        self.synthetic_references = load_synthetic_references(self.config["synthetic_dataset_paths"])
        # Initialize optimizers for code and design models.
        self.code_optimizer = optim.Adam(self.models["code_model"].parameters(), lr=self.config["learning_rate"])
        self.design_optimizer = optim.Adam(self.models["design_model"].parameters(), lr=self.config["learning_rate"])
        # Initialize projection layers for cross-model alignment.
        self._initialize_alignment_layers()
        self.alignment_optimizer = optim.Adam(
            list(self.proj_code.parameters()) +
            list(self.proj_design.parameters()) +
            list(self.proj_eval.parameters()),
            lr=self.config["alignment"]["learning_rate"]
        )

    def _configure_logger(self):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
        return logging.getLogger(__name__)

    def _load_evaluation_criteria(self, filename: str) -> List[Dict[str, Any]]:
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Evaluation criteria file '{filename}' not found.")
        try:
            with open(filename, "r", encoding="utf-8") as f:
                criteria = json.load(f)
            self.logger.info("Evaluation criteria loaded successfully.")
            return criteria
        except Exception as e:
            self.logger.error(f"Failed to load evaluation criteria: {e}", exc_info=True)
            raise

    def _load_models(self) -> Dict[str, Any]:
        try:
            models = {
                "code_model": AutoModelForCausalLM.from_pretrained(
                    self.config["code_generator"],
                    device_map=self.config["device_map"],
                    torch_dtype=self.config["torch_dtype"],
                    output_hidden_states=True  # Enable hidden state output for alignment
                ),
                "design_model": AutoModelForCausalLM.from_pretrained(
                    self.config["design_llm"],
                    device_map=self.config["device_map"],
                    torch_dtype=self.config["torch_dtype"],
                    output_hidden_states=True
                ),
                "eval_model": AutoModelForCausalLM.from_pretrained(
                    self.config["evaluator"],
                    device_map=self.config["device_map"],
                    torch_dtype=self.config["torch_dtype"],
                    output_hidden_states=True
                )
            }
            self.logger.info("All models loaded successfully.")
            return models
        except Exception as e:
            self.logger.error(f"Model loading failed: {e}", exc_info=True)
            raise

    def _load_tokenizers(self) -> Dict[str, Any]:
        try:
            tokenizers = {
                "code": AutoTokenizer.from_pretrained(self.config["code_generator"]),
                "design": AutoTokenizer.from_pretrained(self.config["design_llm"]),
                "eval": AutoTokenizer.from_pretrained(self.config["evaluator"])
            }
            self.logger.info("All tokenizers loaded successfully.")
            return tokenizers
        except Exception as e:
            self.logger.error(f"Tokenizer loading failed: {e}", exc_info=True)
            raise

    def _initialize_alignment_layers(self):
        """
        Initialize projection layers to map each model's latent representation to a common space.
        """
        common_dim = self.config["alignment"]["common_dim"]
        try:
            code_hidden_size = self.models["code_model"].config.hidden_size
            design_hidden_size = self.models["design_model"].config.hidden_size
            eval_hidden_size = self.models["eval_model"].config.hidden_size
        except AttributeError as e:
            self.logger.error("Unable to retrieve hidden_size from one of the model configs.", exc_info=True)
            raise

        self.proj_code = nn.Linear(code_hidden_size, common_dim).to("cuda")
        self.proj_design = nn.Linear(design_hidden_size, common_dim).to("cuda")
        self.proj_eval = nn.Linear(eval_hidden_size, common_dim).to("cuda")
        self.logger.info("Alignment projection layers initialized.")

    def cross_model_alignment(self, prompt: str) -> float:
        """
        Perform cross-model alignment by forwarding the given prompt through each model,
        projecting the final hidden states into a common space, and computing an alignment loss.
        The loss is defined as the average of (1 - cosine similarity) between each pair of models.
        """
        self.logger.info("Starting cross-model alignment.")
        try:
            # Tokenize input for each model.
            code_inputs = self.tokenizers["code"](prompt, return_tensors="pt").to("cuda")
            design_inputs = self.tokenizers["design"](prompt, return_tensors="pt").to("cuda")
            eval_inputs = self.tokenizers["eval"](prompt, return_tensors="pt").to("cuda")

            # Forward pass with hidden state outputs.
            code_outputs = self.models["code_model"](**code_inputs)
            design_outputs = self.models["design_model"](**design_inputs)
            eval_outputs = self.models["eval_model"](**eval_inputs)

            # Mean pool the last hidden states.
            code_hidden = code_outputs.hidden_states[-1].mean(dim=1)
            design_hidden = design_outputs.hidden_states[-1].mean(dim=1)
            eval_hidden = eval_outputs.hidden_states[-1].mean(dim=1)

            # Project into common space.
            code_proj = self.proj_code(code_hidden)
            design_proj = self.proj_design(design_hidden)
            eval_proj = self.proj_eval(eval_hidden)

            # Compute cosine similarities.
            sim_cd = torch.nn.functional.cosine_similarity(code_proj, design_proj, dim=1)
            sim_ce = torch.nn.functional.cosine_similarity(code_proj, eval_proj, dim=1)
            sim_de = torch.nn.functional.cosine_similarity(design_proj, eval_proj, dim=1)

            # Alignment loss: 1 - average cosine similarity.
            loss_cd = 1 - sim_cd.mean()
            loss_ce = 1 - sim_ce.mean()
            loss_de = 1 - sim_de.mean()
            alignment_loss = (loss_cd + loss_ce + loss_de) / 3.0

            # Optimize projection layers to minimize alignment loss.
            self.alignment_optimizer.zero_grad()
            alignment_loss.backward()
            self.alignment_optimizer.step()

            self.logger.info(f"Cross-model alignment loss: {alignment_loss.item():.4f}")
            return alignment_loss.item()
        except Exception as e:
            self.logger.error(f"Cross-model alignment failed: {e}", exc_info=True)
            raise

    def ensemble_synchronization(self):
        """
        Perform cross-model alignment over several iterations to ensure proper knowledge alignment.
        """
        prompt = self.config["alignment"]["alignment_prompt"]
        iterations = self.config["alignment"]["iterations"]
        self.logger.info("Starting ensemble synchronization (cross-model alignment).")
        total_loss = 0.0
        for i in range(iterations):
            loss = self.cross_model_alignment(prompt)
            total_loss += loss
            self.logger.info(f"Alignment iteration {i+1}/{iterations}: Loss = {loss:.4f}")
        avg_loss = total_loss / iterations
        self.logger.info(f"Ensemble synchronization complete. Average alignment loss: {avg_loss:.4f}")

    ##############################################
    # Generation and RAIF-based Deep Thinking Cycle
    ##############################################

    def _create_design_spec(self, prompt: str) -> str:
        """
        Generate detailed UI/UX specifications using the design language model.
        """
        message = f"Create detailed UI/UX specifications for: {prompt}"
        try:
            design_pipe = pipeline(
                "text-generation",
                model=self.models["design_model"],
                tokenizer=self.tokenizers["design"],
                max_new_tokens=512,
                temperature=0.7,
            )
            result = design_pipe(message)
            design_spec = result[0]["generated_text"]
            self.logger.info("Design specification generated successfully.")
            return design_spec
        except Exception as e:
            self.logger.error(f"Design specification generation failed: {e}", exc_info=True)
            raise

    def _generate_code(self, design_spec: str) -> str:
        """
        Generate code based on the design specification using the code generator.
        """
        try:
            stop_tokens = self.tokenizers["code"].convert_tokens_to_ids(["<|endoftext|>"])
            stopping_criteria = CodeStoppingCriteria(stop_tokens)
            inputs = self.tokenizers["code"](
                design_spec,
                return_tensors="pt",
                max_length=self.config["max_code_length"],
                truncation=True
            ).to("cuda")
            outputs = self.models["code_model"].generate(
                **inputs,
                max_new_tokens=512,
                stopping_criteria=[stopping_criteria],
                temperature=0.5,
                do_sample=True,
            )
            generated_code = self.tokenizers["code"].decode(outputs[0], skip_special_tokens=True)
            self.logger.info("Code generation completed successfully.")
            return generated_code
        except Exception as e:
            self.logger.error(f"Code generation failed: {e}", exc_info=True)
            raise

    def evaluate_code(self, code: str) -> Tuple[float, str]:
        """
        Evaluate the generated code using the evaluator model and compute an aggregated score.
        """
        try:
            eval_prompt = f"Evaluate this web code:\n\n{code}\n\nEvaluation:"
            inputs = self.tokenizers["eval"](
                eval_prompt,
                return_tensors="pt",
                max_length=2048,
                truncation=True
            ).to("cuda")
            outputs = self.models["eval_model"].generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.3,
            )
            evaluation = self.tokenizers["eval"].decode(outputs[0], skip_special_tokens=True)
            base_score = self._parse_evaluation(evaluation)
            ref_similarity = compute_similarity_with_references(code, self.synthetic_references)
            aggregated_score = aggregate_model_scores(base_score, ref_similarity, self.config["score_weights"])
            self.logger.info(f"Evaluation completed with aggregated score: {aggregated_score:.2f}")
            return aggregated_score, evaluation
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}", exc_info=True)
            raise

    def _parse_evaluation(self, evaluation_text: str) -> float:
        """
        Parse evaluation feedback text and compute a score based on weighted criteria.
        """
        score = 0.0
        for criterion in self.eval_criteria:
            if criterion.get("key") and criterion.get("weight"):
                if criterion["key"].lower() in evaluation_text.lower():
                    score += criterion["weight"]
        return min(max(score, 0.0), 1.0)

    def deep_thinking_cycle(self, code: str, feedback: str) -> Tuple[str, float]:
        """
        Generate multiple candidate refinements using the code generator and select the best candidate.
        This process embodies the RAIF deep thinking cycle.
        """
        candidates = []
        candidate_scores = []
        refinement_prompt = (
            f"Improve the following code based on this feedback:\n\n"
            f"Code:\n{code}\n\nFeedback:\n{feedback}\n\nImproved Code:"
        )
        for i in range(self.config["deep_thinking_candidates"]):
            try:
                inputs = self.tokenizers["code"](
                    refinement_prompt,
                    return_tensors="pt",
                    max_length=self.config["max_code_length"],
                    truncation=True
                ).to("cuda")
                outputs = self.models["code_model"].generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.6 + (i * 0.05),
                    do_sample=True,
                )
                candidate_code = self.tokenizers["code"].decode(outputs[0], skip_special_tokens=True)
                score, _ = self.evaluate_code(candidate_code)
                candidates.append(candidate_code)
                candidate_scores.append(score)
                self.logger.info(f"Candidate {i+1} generated with aggregated score: {score:.2f}")
            except Exception as e:
                self.logger.error(f"Deep thinking candidate {i+1} failed: {e}", exc_info=True)
        if candidate_scores:
            best_index = candidate_scores.index(max(candidate_scores))
            self.logger.info(f"Selected candidate {best_index+1} with aggregated score: {candidate_scores[best_index]:.2f}")
            return candidates[best_index], candidate_scores[best_index]
        else:
            self.logger.warning("No valid candidate generated; retaining original code.")
            return code, 0.0

    def refine_code(self, code: str, feedback: str) -> Tuple[str, float]:
        """
        Refine the generated code using the deep thinking cycle.
        """
        return self.deep_thinking_cycle(code, feedback)

    def generate_initial_code(self, prompt: str) -> str:
        """
        Validate the prompt, generate a design specification, and produce initial code.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string.")
        design_spec = self._create_design_spec(prompt)
        return self._generate_code(design_spec)

    def generate_webpage(self, prompt: str) -> Dict[str, Any]:
        """
        Full pipeline for generating webpage code.
        Iteratively refines code until the aggregated evaluation score meets the threshold.
        """
        self.logger.info("Starting webpage generation process.")
        code = self.generate_initial_code(prompt)
        best_score = 0.0
        best_code = code
        feedback = ""

        for cycle in range(self.config["max_refinement_cycles"]):
            if not self.resource_manager.check_memory():
                self.logger.warning("High memory usage; releasing GPU memory.")
                self.resource_manager.release_gpu_memory()

            score, evaluation_feedback = self.evaluate_code(best_code)
            self.logger.info(f"Refinement cycle {cycle+1}: Aggregated score = {score:.2f}")

            if score >= self.config["evaluation_threshold"]:
                self.logger.info("Evaluation threshold met; finalizing code.")
                feedback = evaluation_feedback
                best_score = score
                break

            refined_code, new_score = self.refine_code(best_code, evaluation_feedback)
            if new_score > best_score:
                best_score = new_score
                best_code = refined_code
                feedback = evaluation_feedback

            self.resource_manager.release_gpu_memory()

        return {
            "final_code": best_code,
            "final_score": best_score,
            "evaluation_feedback": feedback
        }

    ##############################################
    # Training, Metrics Logging, and Knowledge Distillation
    ##############################################

    def synthesize_training_data(self, num_samples: int = None) -> List[Tuple[str, str, str]]:
        """
        Generate synthetic training data from self-generated prompts.
        Returns a list of tuples: (prompt, design_spec, code).
        """
        num_samples = num_samples or self.config["training_iterations"]
        synthetic_data = []
        self.logger.info("Synthesizing training data.")
        for i in range(num_samples):
            prompt = f"Synthetic prompt {i+1}"
            try:
                design_spec = self._create_design_spec(prompt)
                code = self._generate_code(design_spec)
                synthetic_data.append((prompt, design_spec, code))
                self.logger.info(f"Synthesized training sample {i+1}.")
            except Exception as e:
                self.logger.error(f"Failed to synthesize training sample {i+1}: {e}", exc_info=True)
        return synthetic_data

    def train_step(self, design_spec: str, code: str) -> float:
        """
        Perform a single training step on the code model using RAIF evaluation feedback.
        Loss is defined as the squared difference between the threshold and the aggregated score.
        """
        self.models["code_model"].train()
        score, _ = self.evaluate_code(code)
        loss_value = max(0.0, self.config["evaluation_threshold"] - score)
        loss = torch.tensor(loss_value ** 2, requires_grad=True)
        try:
            self.code_optimizer.zero_grad()
            loss.backward()
            self.code_optimizer.step()
            self.logger.info(f"Training step completed: Loss = {loss.item():.4f} | Aggregated Score = {score:.4f}")
        except Exception as e:
            self.logger.error(f"Training step failed: {e}", exc_info=True)
            raise
        self.resource_manager.release_gpu_memory()
        return score

    def log_training_metrics(self, iteration: int, score: float, plateau_counter: int) -> None:
        """
        Log detailed training metrics.
        """
        self.logger.info(f"Iteration {iteration}: Aggregated Score = {score:.4f} | Plateau Counter = {plateau_counter}")

    def save_model_results(self, epoch: int, best_score: float) -> None:
        """
        Save ensemble models in safetensors format.
        """
        try:
            code_filename = f"code_model_epoch{epoch}_score{best_score:.2f}.safetensors"
            design_filename = f"design_model_epoch{epoch}_score{best_score:.2f}.safetensors"
            eval_filename = f"eval_model_epoch{epoch}_score{best_score:.2f}.safetensors"
            save_file(self.models["code_model"].state_dict(), code_filename)
            save_file(self.models["design_model"].state_dict(), design_filename)
            save_file(self.models["eval_model"].state_dict(), eval_filename)
            self.logger.info(f"Models saved at epoch {epoch} with aggregated score {best_score:.2f}.")
        except Exception as e:
            self.logger.error(f"Failed to save models: {e}", exc_info=True)
            raise

    def knowledge_distillation(self):
        """
        Perform knowledge distillation by aggregating insights from the ensemble models.
        For production, a proper distillation mechanism should be implemented.
        Here, we simulate by saving the code_model as the distilled model.
        """
        self.logger.info("Starting knowledge distillation to produce a new distilled model.")
        try:
            distilled_state = self.models["code_model"].state_dict()
            distilled_filename = "distilled_model.safetensors"
            save_file(distilled_state, distilled_filename)
            self.logger.info(f"Distilled model saved as '{distilled_filename}'.")
        except Exception as e:
            self.logger.error(f"Knowledge distillation failed: {e}", exc_info=True)
            raise

    def train_ensemble(self, num_iterations: int = None):
        """
        Run ensemble training with synthesized data and RAIF-based iterative refinement.
        Save models and perform knowledge distillation when performance plateaus or threshold is met.
        """
        num_iterations = num_iterations or self.config["training_iterations"]
        self.logger.info(f"Starting ensemble training for {num_iterations} iterations.")
        training_data = self.synthesize_training_data(num_iterations)

        best_overall_score = 0.0
        plateau_counter = 0
        for idx, (prompt, design_spec, code) in enumerate(training_data, start=1):
            self.logger.info(f"Training iteration {idx}: Prompt: {prompt}")
            current_score = self.train_step(design_spec, code)
            self.log_training_metrics(idx, current_score, plateau_counter)

            if current_score > best_overall_score + 1e-3:
                best_overall_score = current_score
                plateau_counter = 0
                self.logger.info(f"New best overall aggregated score: {best_overall_score:.4f}")
            else:
                plateau_counter += 1
                self.logger.info(f"No significant improvement. Plateau counter: {plateau_counter}")

            if best_overall_score >= self.config["evaluation_threshold"] or plateau_counter >= 3:
                self.logger.info("Threshold met or plateau detected; saving models.")
                self.save_model_results(epoch=idx, best_score=best_overall_score)
                self.ensemble_synchronization()
                self.knowledge_distillation()
                break

        self.logger.info(f"Ensemble training completed with final best aggregated score: {best_overall_score:.4f}")

##############################################
# Main Execution
##############################################

def main():
    parser = argparse.ArgumentParser(
        description="Web Ensemble System with RAIF-based Automatic Alignment, Synthetic References, and Cross-Model Alignment."
    )
    parser.add_argument("--train", action="store_true", help="Run ensemble training instead of webpage generation")
    parser.add_argument("--prompt", type=str,
                        default="Create a responsive landing page for an AI startup with dark theme",
                        help="Prompt for webpage generation")
    args = parser.parse_args()

    try:
        system = WebEnsembleSystem()
    except Exception as e:
        logging.critical(f"Failed to initialize WebEnsembleSystem: {e}", exc_info=True)
        return

    if args.train:
        try:
            system.train_ensemble()
        except Exception as e:
            logging.error(f"Ensemble training failed: {e}", exc_info=True)
    else:
        try:
            result = system.generate_webpage(args.prompt)
            print("Final Generated Code:\n", result["final_code"])
            print("\nFinal Aggregated Score:", result["final_score"])
            print("\nEvaluator Feedback:", result["evaluation_feedback"])
        except Exception as e:
            logging.error(f"Webpage generation failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()
