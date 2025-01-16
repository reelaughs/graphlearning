import torch
from pykeen.datasets import WN18RR
from pykeen.models import TransE, TransR, DistMult
from pykeen.training import SLCWATrainingLoop
from pykeen.evaluation import RankBasedEvaluator

# Load the dataset
dataset = WN18RR()
training_triples_factory = dataset.training

# Set a random seed globally for reproducibility
torch.manual_seed(42)

def train_and_evaluate(model_class, model_name):
    # Pick the model
    model = model_class(triples_factory=training_triples_factory)

    # Pick an optimizer from Torch
    optimizer = torch.optim.Adam(params=model.get_grad_params())

    # Pick a training approach (sLCWA)
    training_loop = SLCWATrainingLoop(
        model=model,
        triples_factory=training_triples_factory,
        optimizer=optimizer,
    )

    # Train the model
    training_loop.train(
        triples_factory=training_triples_factory,  # Pass the triples_factory here
        num_epochs=3,
        batch_size=128,
    )

    # Pick an evaluator
    evaluator = RankBasedEvaluator()

    # Evaluate the model
    results = evaluator.evaluate(
        model=model,
        mapped_triples=dataset.testing.mapped_triples,
        batch_size=1024,
        additional_filter_triples=[
            dataset.training.mapped_triples,
            dataset.validation.mapped_triples,
        ],
    )

    print("Evaluation Results:")
    print(f"Mean Rank: {results.get_metric('mean_rank')}")
    print(f"Mean Reciprocal Rank (MRR): {results.get_metric('mean_reciprocal_rank')}")
    print(f"Hits@1: {results.get_metric('hits@1')}")
    print(f"Hits@3: {results.get_metric('hits@3')}")
    print(f"Hits@10: {results.get_metric('hits@10')}")

# Train and evaluate TransE
train_and_evaluate(TransE, "TransE")

# Train and evaluate TransR
train_and_evaluate(TransR, "TransR")

# Train and evaluate DistMult
train_and_evaluate(DistMult, "DistMult")

