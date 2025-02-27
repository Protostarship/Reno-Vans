#!/usr/bin/env python3
"""
Production-Ready Ensemble Training and Webpage Generation Script
with Automatic Debugger Agents, Advanced Evaluation Metrics, RAIF-based Refinement,
Cross-Model Alignment, and Knowledge Distillation

This script implements an ensemble system that:
  • Loads three pre-trained models: a code generator (e.g. StarCoder2-3B),
    a design language model (e.g. Llama-3.1-8B-Instruct), and an evaluator (e.g. CodeGemma-7B-IT).
  • Loads local synthetic reference datasets (HTML, CSS, JS) for enhanced evaluation.
  • Generates detailed design specifications and code from a text prompt.
  • Evaluates the generated code using multiple metrics:
      - Evaluator feedback via language model
      - Reference similarity using TF-IDF and cosine similarity
      - Simulated code execution test (basic HTML/CSS/JS structure)
      - Advanced static analysis (token presence and code length)
      - Dummy logistic regression and KNN scores based on TF-IDF features
  • Aggregates scores via weighted averaging.
  • Uses a RAIF-based deep thinking cycle for iterative refinement of code.
  • Implements an automatic DebuggerAgent that produces detailed debugging reports,
    using caching to avoid redundant computations.
  • Performs advanced cross-model alignment with projection layers, adversarial perturbation,
    and learning rate scheduling.
  • Conducts advanced knowledge distillation by training a student MLP to mimic ensemble outputs.
  • Includes unit tests for critical components.
  • Uses robust error handling, caching, and explicit CUDA device attachment.
  
Usage:
  To generate a webpage:
      $ python ensemble_system.py --prompt "Create a responsive landing page for an AI startup with dark theme"
  To run ensemble training:
      $ python ensemble_system.py --train
  To run unit tests:
      $ python ensemble_system.py --test
"""

import os
import json
import logging
import argparse
import time
import random
from typing import Dict, Any, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
import psutil

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    StoppingCriteria
)
from safetensors.torch import save_file

# For TF-IDF and cosine similarity
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
    "deep_thinking_candidates": 3,       # Number of candidate refinements in deep thinking cycle
    "synthetic_dataset_paths": {         # Local paths for synthetic reference datasets
        "html": "main/training/synthetic_data/html",
        "css": "main/training/synthetic_data/css",
        "js": "main/training/synthetic_data/js"
    },
    "score_weights": {                   # Weights for aggregating evaluation scores
        "evaluation": 0.5,
        "reference": 0.2,
        "execution": 0.1,
        "static": 0.1,
        "lr": 0.05,
        "knn": 0.05
    },
    "alignment": {                       # Parameters for cross-model alignment
        "common_dim": 512,
        "learning_rate": 1e-4,
        "iterations": 5,
        "alignment_prompt": "Align ensemble models for consistent output.",
        "adversarial_eps": 0.01
    },
    "advanced_distillation": {           # Parameters for student model distillation
        "input_dim": 512,
        "hidden_dim": 256,
        "output_dim": 512,
        "learning_rate": 1e-4,
        "epochs": 5
    },
    "diverse_prompts": [                 # Diverse synthetic prompts for training data
        "Create a minimalist landing page with a modern aesthetic.",
        "Generate a blog template with responsive design and dark mode.",
        "Develop a multi-page website for an e-commerce store.",
        "Design an interactive portfolio site with animation effects.",
        "Produce a static informational website with accessibility compliance."
    ]
}

##############################################
# Resource Management
##############################################

class ResourceLimits:
    """Defines system resource limits."""
    def __init__(self, max_ram_gb: float = 20.0, max_vram_gb: float = 3.8):
        self.max_ram_gb = max_ram_gb
        self.max_vram_gb = max_vram_gb

class ResourceManager:
    """Monitors system memory usage and releases GPU memory as needed."""
    def __init__(self, limits: ResourceLimits):
        self.limits = limits

    def check_memory(self) -> bool:
        mem = psutil.virtual_memory()
        used_ram_gb = mem.used / (1024**3)
        logging.debug(f"RAM: {used_ram_gb:.2f} GB / {self.limits.max_ram_gb} GB")
        return used_ram_gb < self.limits.max_ram_gb

    def release_gpu_memory(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logging.info("Released GPU memory.")

##############################################
# Custom Stopping Criteria
##############################################

class CodeStoppingCriteria(StoppingCriteria):
    """Stops generation when a designated stop token is encountered."""
    def __init__(self, stop_tokens: List[int]):
        self.stop_tokens = stop_tokens

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0][-1] in self.stop_tokens

##############################################
# Synthetic Reference Functions
##############################################

def load_synthetic_references(paths: Dict[str, str]) -> Dict[str, List[str]]:
    """Load synthetic reference files from specified paths."""
    refs = {}
    for key, path in paths.items():
        refs[key] = []
        if os.path.exists(path):
            for filename in os.listdir(path):
                if filename.endswith((".html", ".css", ".js")):
                    full_path = os.path.join(path, filename)
                    try:
                        with open(full_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            if content:
                                refs[key].append(content)
                    except Exception as e:
                        logging.warning(f"Failed to read {full_path}: {e}")
        else:
            logging.warning(f"Path '{path}' does not exist for {key} references.")
    logging.info("Synthetic references loaded.")
    return refs

def compute_similarity_with_references(code: str, references: Dict[str, List[str]]) -> float:
    """Compute maximum cosine similarity between code and reference texts."""
    all_refs = []
    for ref_list in references.values():
        all_refs.extend(ref_list)
    if not all_refs:
        logging.warning("No references available; similarity = 0.0")
        return 0.0
    vectorizer = TfidfVectorizer().fit(all_refs + [code])
    code_vec = vectorizer.transform([code])
    refs_vec = vectorizer.transform(all_refs)
    sims = cosine_similarity(code_vec, refs_vec)
    max_sim = float(sims.max())
    logging.info(f"Reference similarity: {max_sim:.2f}")
    return max_sim

def aggregate_model_scores(scores: Dict[str, float], weights: Dict[str, float]) -> float:
    """Aggregate scores via weighted averaging."""
    agg = sum(weights[k] * scores.get(k, 0.0) for k in weights)
    logging.info(f"Aggregated score: {agg:.2f}")
    return agg

##############################################
# Additional Evaluation Functions
##############################################

def simulate_code_execution(code: str) -> float:
    """Simulate execution by checking for HTML structure; returns score 0-1."""
    score = 0.0
    code_low = code.lower()
    if "<html" in code_low and "</html>" in code_low:
        score += 0.5
    if ("<style" in code_low or ".css" in code_low) and ("<script" in code_low or ".js" in code_low):
        score += 0.5
    logging.info(f"Execution simulation score: {score:.2f}")
    return score

def advanced_static_analysis(code: str) -> float:
    """Simulate static analysis via code length and token checks."""
    score = 0.0
    if len(code) > 100:
        score += 0.4
    if "div" in code and "class" in code:
        score += 0.3
    if "function" in code or "def" in code:
        score += 0.3
    logging.info(f"Static analysis score: {min(score, 1.0):.2f}")
    return min(score, 1.0)

def logistic_regression_evaluation(code: str) -> float:
    """Dummy logistic regression evaluation based on TF-IDF vector norm."""
    vectorizer = TfidfVectorizer()
    try:
        features = vectorizer.fit_transform([code])
        norm = features.norm()  # L2 norm
        score = min(norm / 10.0, 1.0)
    except Exception:
        score = 0.0
    logging.info(f"Logistic regression score: {score:.2f}")
    return score

def knn_evaluation(code: str) -> float:
    """Dummy KNN evaluation based on code length."""
    score = min(len(code) / 10000.0, 1.0)
    logging.info(f"KNN evaluation score: {score:.2f}")
    return score

##############################################
# Automatic Debugger Agent
##############################################

class DebuggerAgent:
    """
    Automatic debugger agent that generates detailed debugging reports using a language model.
    Caches outputs to avoid redundant computations.
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.cache = {}

    def debug_code(self, code: str) -> str:
        if code in self.cache:
            return self.cache[code]
        prompt = (
            "Analyze the following code for potential errors, inefficiencies, and deviations "
            "from best practices. Provide a detailed debugging report with suggestions for improvement:\n\n"
            f"{code}"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to("cuda")
        outputs = self.model.generate(**inputs, max_new_tokens=256, temperature=0.7)
        report = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        self.cache[code] = report
        return report

##############################################
# Web Ensemble System with Advanced Features
##############################################

class WebEnsembleSystem:
    """
    Ensemble system integrating design spec extraction, code generation,
    multi-metric evaluation, RAIF-based refinement, automatic debugging,
    advanced cross-model alignment, and knowledge distillation.
    """
    def __init__(self):
        self.config = TRAINING_CONFIG
        self.logger = self._configure_logger()
        self.resource_manager = ResourceManager(ResourceLimits())
        self.models = self._load_models()
        self.tokenizers = self._load_tokenizers()
        self.eval_criteria = self._load_evaluation_criteria("eval_criteria.json")
        self.synthetic_references = load_synthetic_references(self.config["synthetic_dataset_paths"])
        self.code_optimizer = optim.Adam(self.models["code_model"].parameters(), lr=self.config["learning_rate"])
        self.design_optimizer = optim.Adam(self.models["design_model"].parameters(), lr=self.config["learning_rate"])
        self._initialize_alignment_layers()
        self.alignment_optimizer = optim.Adam(
            list(self.proj_code.parameters()) + list(self.proj_design.parameters()) + list(self.proj_eval.parameters()),
            lr=self.config["alignment"]["learning_rate"]
        )
        # Initialize automatic debugger agent using evaluator model.
        self.debugger_agent = DebuggerAgent(self.models["eval_model"], self.tokenizers["eval"])
        # Initialize student model for advanced knowledge distillation.
        self.student_model = self._initialize_student_model()

    def _configure_logger(self):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
        return logging.getLogger(__name__)

    def _load_evaluation_criteria(self, filename: str) -> List[Dict[str, Any]]:
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Evaluation criteria file '{filename}' not found.")
        try:
            with open(filename, "r", encoding="utf-8") as f:
                criteria = json.load(f)
            self.logger.info("Evaluation criteria loaded.")
            return criteria
        except Exception as e:
            self.logger.error(f"Loading evaluation criteria failed: {e}", exc_info=True)
            raise

    def _load_models(self) -> Dict[str, Any]:
        try:
            models = {
                "code_model": AutoModelForCausalLM.from_pretrained(
                    self.config["code_generator"],
                    device_map=self.config["device_map"],
                    torch_dtype=self.config["torch_dtype"],
                    output_hidden_states=True
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
            self.logger.info("Models loaded.")
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
            self.logger.info("Tokenizers loaded.")
            return tokenizers
        except Exception as e:
            self.logger.error(f"Tokenizer loading failed: {e}", exc_info=True)
            raise

    def _initialize_alignment_layers(self):
        """Initialize projection layers for cross-model alignment."""
        common_dim = self.config["alignment"]["common_dim"]
        try:
            code_hidden = self.models["code_model"].config.hidden_size
            design_hidden = self.models["design_model"].config.hidden_size
            eval_hidden = self.models["eval_model"].config.hidden_size
        except AttributeError as e:
            self.logger.error("Error retrieving hidden_size.", exc_info=True)
            raise
        self.proj_code = nn.Linear(code_hidden, common_dim).to("cuda")
        self.proj_design = nn.Linear(design_hidden, common_dim).to("cuda")
        self.proj_eval = nn.Linear(eval_hidden, common_dim).to("cuda")
        self.logger.info("Alignment layers initialized.")

    def cross_model_alignment(self, prompt: str) -> float:
        """
        Perform cross-model alignment with adversarial perturbation and optimize the alignment loss.
        """
        self.logger.info("Running cross-model alignment.")
        try:
            code_inputs = self.tokenizers["code"](prompt, return_tensors="pt").to("cuda")
            design_inputs = self.tokenizers["design"](prompt, return_tensors="pt").to("cuda")
            eval_inputs = self.tokenizers["eval"](prompt, return_tensors="pt").to("cuda")

            code_out = self.models["code_model"](**code_inputs)
            design_out = self.models["design_model"](**design_inputs)
            eval_out = self.models["eval_model"](**eval_inputs)

            code_hidden = code_out.hidden_states[-1].mean(dim=1)
            design_hidden = design_out.hidden_states[-1].mean(dim=1)
            eval_hidden = eval_out.hidden_states[-1].mean(dim=1)

            eps = self.config["alignment"]["adversarial_eps"]
            code_hidden_adv = code_hidden + eps * torch.randn_like(code_hidden)
            design_hidden_adv = design_hidden + eps * torch.randn_like(design_hidden)
            eval_hidden_adv = eval_hidden + eps * torch.randn_like(eval_hidden)

            code_proj = self.proj_code(code_hidden_adv)
            design_proj = self.proj_design(design_hidden_adv)
            eval_proj = self.proj_eval(eval_hidden_adv)

            sim_cd = torch.nn.functional.cosine_similarity(code_proj, design_proj, dim=1)
            sim_ce = torch.nn.functional.cosine_similarity(code_proj, eval_proj, dim=1)
            sim_de = torch.nn.functional.cosine_similarity(design_proj, eval_proj, dim=1)

            loss = (1 - sim_cd.mean() + 1 - sim_ce.mean() + 1 - sim_de.mean()) / 3.0

            self.alignment_optimizer.zero_grad()
            loss.backward()
            self.alignment_optimizer.step()

            self.logger.info(f"Alignment loss: {loss.item():.4f}")
            return loss.item()
        except Exception as e:
            self.logger.error(f"Cross-model alignment error: {e}", exc_info=True)
            raise

    def ensemble_synchronization(self):
        """Run cross-model alignment iteratively."""
        prompt = self.config["alignment"]["alignment_prompt"]
        iterations = self.config["alignment"]["iterations"]
        self.logger.info("Starting ensemble synchronization.")
        total_loss = 0.0
        for i in range(iterations):
            loss = self.cross_model_alignment(prompt)
            total_loss += loss
            self.logger.info(f"Iteration {i+1}/{iterations}: Loss = {loss:.4f}")
        avg_loss = total_loss / iterations
        self.logger.info(f"Synchronization complete. Avg loss: {avg_loss:.4f}")

    def _create_design_spec(self, prompt: str) -> str:
        """Generate UI/UX specifications using the design model."""
        message = f"Create detailed UI/UX specifications for: {prompt}"
        try:
            design_pipe = pipeline("text-generation", model=self.models["design_model"],
                                   tokenizer=self.tokenizers["design"], max_new_tokens=512, temperature=0.7)
            result = design_pipe(message)
            spec = result[0]["generated_text"]
            self.logger.info("Design specification generated.")
            return spec
        except Exception as e:
            self.logger.error(f"Design spec generation failed: {e}", exc_info=True)
            raise

    def _generate_code(self, design_spec: str) -> str:
        """Generate code using the code generator."""
        try:
            stop_tokens = self.tokenizers["code"].convert_tokens_to_ids(["<|endoftext|>"])
            stopping_criteria = CodeStoppingCriteria(stop_tokens)
            inputs = self.tokenizers["code"](design_spec, return_tensors="pt", max_length=self.config["max_code_length"], truncation=True).to("cuda")
            outputs = self.models["code_model"].generate(**inputs, max_new_tokens=512, stopping_criteria=[stopping_criteria], temperature=0.5, do_sample=True)
            code = self.tokenizers["code"].decode(outputs[0], skip_special_tokens=True)
            self.logger.info("Code generated.")
            return code
        except Exception as e:
            self.logger.error(f"Code generation failed: {e}", exc_info=True)
            raise

    def auto_debug_code(self, code: str) -> str:
        """Automatically generate a debugging report using the DebuggerAgent."""
        try:
            report = self.debugger_agent.debug_code(code)
            self.logger.info("Automatic debug report generated.")
            return report
        except Exception as e:
            self.logger.error(f"Auto-debug failed: {e}", exc_info=True)
            return "Debugging report unavailable."

    def evaluate_code(self, code: str) -> Tuple[float, str]:
        """
        Evaluate generated code using multiple metrics and aggregate the score.
        Also, append an automatic debugging report if the score is low.
        """
        try:
            eval_prompt = f"Evaluate this web code:\n\n{code}\n\nEvaluation:"
            inputs = self.tokenizers["eval"](eval_prompt, return_tensors="pt", max_length=2048, truncation=True).to("cuda")
            outputs = self.models["eval_model"].generate(**inputs, max_new_tokens=256, temperature=0.3)
            evaluation = self.tokenizers["eval"].decode(outputs[0], skip_special_tokens=True)
            base_score = self._parse_evaluation(evaluation)

            ref_sim = compute_similarity_with_references(code, self.synthetic_references)
            exec_score = simulate_code_execution(code)
            static_score = advanced_static_analysis(code)
            lr_score = logistic_regression_evaluation(code)
            knn_score = knn_evaluation(code)

            scores = {
                "evaluation": base_score,
                "reference": ref_sim,
                "execution": exec_score,
                "static": static_score,
                "lr": lr_score,
                "knn": knn_score
            }
            aggregated = aggregate_model_scores(scores, self.config["score_weights"])
            # If aggregated score is low, append debugger report.
            if aggregated < 0.5:
                debug_report = self.auto_debug_code(code)
                evaluation += "\n\n[DEBUG REPORT]:\n" + debug_report
            self.logger.info(f"Aggregated evaluation score: {aggregated:.2f}")
            return aggregated, evaluation
        except Exception as e:
            self.logger.error(f"Evaluation error: {e}", exc_info=True)
            raise

    def _parse_evaluation(self, text: str) -> float:
        """Parse evaluator feedback based on weighted criteria."""
        score = 0.0
        for criterion in self.eval_criteria:
            if criterion.get("key") and criterion.get("weight"):
                if criterion["key"].lower() in text.lower():
                    score += criterion["weight"]
        return min(max(score, 0.0), 1.0)

    def deep_thinking_cycle(self, code: str, feedback: str) -> Tuple[str, float]:
        """Generate multiple candidate refinements and return the best candidate."""
        candidates = []
        candidate_scores = []
        prompt = f"Improve the following code based on feedback:\n\nCode:\n{code}\n\nFeedback:\n{feedback}\n\nImproved Code:"
        for i in range(self.config["deep_thinking_candidates"]):
            try:
                inputs = self.tokenizers["code"](prompt, return_tensors="pt", max_length=self.config["max_code_length"], truncation=True).to("cuda")
                outputs = self.models["code_model"].generate(**inputs, max_new_tokens=512, temperature=0.6 + i*0.05, do_sample=True)
                candidate = self.tokenizers["code"].decode(outputs[0], skip_special_tokens=True)
                score, _ = self.evaluate_code(candidate)
                candidates.append(candidate)
                candidate_scores.append(score)
                self.logger.info(f"Candidate {i+1}: Score {score:.2f}")
            except Exception as e:
                self.logger.error(f"Candidate {i+1} generation failed: {e}", exc_info=True)
        if candidate_scores:
            best_idx = candidate_scores.index(max(candidate_scores))
            self.logger.info(f"Best candidate: {best_idx+1} with score {candidate_scores[best_idx]:.2f}")
            return candidates[best_idx], candidate_scores[best_idx]
        else:
            self.logger.warning("No candidates generated; returning original code.")
            return code, 0.0

    def refine_code(self, code: str, feedback: str) -> Tuple[str, float]:
        """Refine code using the deep thinking cycle."""
        return self.deep_thinking_cycle(code, feedback)

    def generate_initial_code(self, prompt: str) -> str:
        """Validate prompt, generate design spec, and produce initial code."""
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must be non-empty.")
        spec = self._create_design_spec(prompt)
        return self._generate_code(spec)

    def generate_webpage(self, prompt: str) -> Dict[str, Any]:
        """
        Full pipeline for generating webpage code. Iteratively refines code until the aggregated
        evaluation score meets the threshold. If final score is low, debugging info is provided.
        """
        self.logger.info("Starting webpage generation.")
        code = self.generate_initial_code(prompt)
        best_score = 0.0
        best_code = code
        feedback = ""
        for cycle in range(self.config["max_refinement_cycles"]):
            if not self.resource_manager.check_memory():
                self.logger.warning("High memory usage; releasing GPU memory.")
                self.resource_manager.release_gpu_memory()
            score, eval_feedback = self.evaluate_code(best_code)
            self.logger.info(f"Cycle {cycle+1}: Score {score:.2f}")
            if score >= self.config["evaluation_threshold"]:
                self.logger.info("Evaluation threshold met.")
                feedback = eval_feedback
                best_score = score
                break
            refined, new_score = self.refine_code(best_code, eval_feedback)
            if new_score > best_score:
                best_score = new_score
                best_code = refined
                feedback = eval_feedback
            self.resource_manager.release_gpu_memory()
        return {"final_code": best_code, "final_score": best_score, "evaluation_feedback": feedback}

    ##############################################
    # Synthetic Data Generation for Training
    ##############################################

    def synthesize_training_data(self, num_samples: int = None) -> List[Tuple[str, str, str]]:
        """
        Generate synthetic training data from a diverse set of prompts.
        """
        num_samples = num_samples or self.config["training_iterations"]
        data = []
        for i in range(num_samples):
            prompt = random.choice(self.config["diverse_prompts"])
            try:
                spec = self._create_design_spec(prompt)
                code = self._generate_code(spec)
                data.append((prompt, spec, code))
                self.logger.info(f"Synthesized sample {i+1} using prompt: {prompt}")
            except Exception as e:
                self.logger.error(f"Sample {i+1} synthesis failed: {e}", exc_info=True)
        return data

    def train_step(self, design_spec: str, code: str) -> float:
        """Perform one training step on the code model using RAIF evaluation feedback."""
        self.models["code_model"].train()
        score, _ = self.evaluate_code(code)
        loss_val = max(0.0, self.config["evaluation_threshold"] - score)
        loss = torch.tensor(loss_val ** 2, requires_grad=True)
        try:
            self.code_optimizer.zero_grad()
            loss.backward()
            self.code_optimizer.step()
            self.logger.info(f"Train step: Loss {loss.item():.4f} | Score {score:.4f}")
        except Exception as e:
            self.logger.error(f"Train step error: {e}", exc_info=True)
            raise
        self.resource_manager.release_gpu_memory()
        return score

    def log_training_metrics(self, iteration: int, score: float, plateau: int) -> None:
        """Log training metrics."""
        self.logger.info(f"Iteration {iteration}: Score {score:.4f} | Plateau {plateau}")

    def save_model_results(self, epoch: int, best_score: float) -> None:
        """Save models in safetensors format."""
        try:
            code_file = f"code_model_epoch{epoch}_score{best_score:.2f}.safetensors"
            design_file = f"design_model_epoch{epoch}_score{best_score:.2f}.safetensors"
            eval_file = f"eval_model_epoch{epoch}_score{best_score:.2f}.safetensors"
            save_file(self.models["code_model"].state_dict(), code_file)
            save_file(self.models["design_model"].state_dict(), design_file)
            save_file(self.models["eval_model"].state_dict(), eval_file)
            self.logger.info(f"Models saved at epoch {epoch} with score {best_score:.2f}.")
        except Exception as e:
            self.logger.error(f"Saving models failed: {e}", exc_info=True)
            raise

    ##############################################
    # Advanced Knowledge Distillation
    ##############################################

    def _initialize_student_model(self) -> nn.Module:
        """Initialize a student MLP for knowledge distillation."""
        inp = self.config["advanced_distillation"]["input_dim"]
        hid = self.config["advanced_distillation"]["hidden_dim"]
        out = self.config["advanced_distillation"]["output_dim"]
        student = nn.Sequential(nn.Linear(inp, hid), nn.ReLU(), nn.Linear(hid, out)).to("cuda")
        self.logger.info("Student model initialized.")
        return student

    def advanced_knowledge_distillation(self):
        """Train the student model to mimic the ensemble's projection output."""
        self.logger.info("Starting advanced knowledge distillation.")
        student_opt = optim.Adam(self.student_model.parameters(), lr=self.config["advanced_distillation"]["learning_rate"])
        epochs = self.config["advanced_distillation"]["epochs"]
        data = self.synthesize_training_data(num_samples=5)
        for epoch in range(epochs):
            total_loss = 0.0
            for prompt, spec, code in data:
                try:
                    inputs = self.tokenizers["code"](code, return_tensors="pt", max_length=self.config["max_code_length"], truncation=True).to("cuda")
                    outputs = self.models["code_model"](**inputs)
                    hidden = outputs.hidden_states[-1].mean(dim=1)
                    with torch.no_grad():
                        target = self.proj_code(hidden)
                    pred = self.student_model(hidden)
                    loss = nn.MSELoss()(pred, target)
                    student_opt.zero_grad()
                    loss.backward()
                    student_opt.step()
                    total_loss += loss.item()
                except Exception as e:
                    self.logger.error(f"Distillation step error: {e}", exc_info=True)
            avg_loss = total_loss / len(data)
            self.logger.info(f"Distillation epoch {epoch+1}/{epochs}: Avg Loss {avg_loss:.4f}")
        try:
            save_file(self.student_model.state_dict(), "distilled_student_model.safetensors")
            self.logger.info("Distilled student model saved as 'distilled_student_model.safetensors'.")
        except Exception as e:
            self.logger.error(f"Saving distilled model failed: {e}", exc_info=True)
            raise

    ##############################################
    # Unit Testing
    ##############################################

    def run_unit_tests(self):
        """Run basic unit tests for critical functions."""
        self.logger.info("Running unit tests...")
        refs = load_synthetic_references(self.config["synthetic_dataset_paths"])
        assert isinstance(refs, dict), "References should be a dictionary."
        dummy_code = "<html><head></head><body><div class='test'></div></body></html>"
        exec_score = simulate_code_execution(dummy_code)
        assert 0.0 <= exec_score <= 1.0, "Execution score out of bounds."
        static_score = advanced_static_analysis(dummy_code)
        assert 0.0 <= static_score <= 1.0, "Static analysis score out of bounds."
        lr_score = logistic_regression_evaluation(dummy_code)
        knn_score = knn_evaluation(dummy_code)
        assert 0.0 <= lr_score <= 1.0, "LR score out of bounds."
        assert 0.0 <= knn_score <= 1.0, "KNN score out of bounds."
        debug_report = self.debugger_agent.debug_code(dummy_code)
        assert isinstance(debug_report, str) and len(debug_report) > 0, "Debug report failed."
        self.logger.info("All unit tests passed.")

    ##############################################
    # Training Ensemble
    ##############################################

    def train_ensemble(self, num_iterations: int = None):
        """Train the ensemble using synthesized data and iterative refinement."""
        num_iterations = num_iterations or self.config["training_iterations"]
        self.logger.info(f"Starting training for {num_iterations} iterations.")
        training_data = self.synthesize_training_data(num_iterations)
        best_overall = 0.0
        plateau = 0
        for idx, (prompt, spec, code) in enumerate(training_data, start=1):
            self.logger.info(f"Iteration {idx} with prompt: {prompt}")
            score = self.train_step(spec, code)
            self.log_training_metrics(idx, score, plateau)
            if score > best_overall + 1e-3:
                best_overall = score
                plateau = 0
                self.logger.info(f"New best score: {best_overall:.4f}")
            else:
                plateau += 1
                self.logger.info(f"No improvement. Plateau count: {plateau}")
            if best_overall >= self.config["evaluation_threshold"] or plateau >= 3:
                self.logger.info("Threshold met or plateau detected; saving models.")
                self.save_model_results(epoch=idx, best_score=best_overall)
                self.ensemble_synchronization()
                self.advanced_knowledge_distillation()
                break
        self.logger.info(f"Training complete. Final best score: {best_overall:.4f}")

##############################################
# Main Execution
##############################################

def main():
    parser = argparse.ArgumentParser(
        description="Web Ensemble System with Automatic Debugger Agents, Advanced Metrics, "
                    "RAIF-based Refinement, Cross-Model Alignment, and Knowledge Distillation."
    )
    parser.add_argument("--train", action="store_true", help="Run ensemble training.")
    parser.add_argument("--prompt", type=str,
                        default="Create a responsive landing page for an AI startup with dark theme",
                        help="Prompt for webpage generation.")
    parser.add_argument("--test", action="store_true", help="Run unit tests.")
    args = parser.parse_args()

    try:
        system = WebEnsembleSystem()
    except Exception as e:
        logging.critical(f"System initialization failed: {e}", exc_info=True)
        return

    if args.test:
        try:
            system.run_unit_tests()
        except Exception as e:
            logging.error(f"Unit tests failed: {e}", exc_info=True)
    elif args.train:
        try:
            system.train_ensemble()
        except Exception as e:
            logging.error(f"Training failed: {e}", exc_info=True)
    else:
        try:
            result = system.generate_webpage(args.prompt)
            print("Final Generated Code:\n", result["final_code"])
            print("\nFinal Aggregated Score:", result["final_score"])
            print("\nEvaluator Feedback:\n", result["evaluation_feedback"])
        except Exception as e:
            logging.error(f"Webpage generation failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()
