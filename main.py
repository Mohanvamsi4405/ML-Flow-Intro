"""Run the full pipeline in one go: python main.py"""
from src import data_ingestion, data_transformation, model_evaluation, model_trainer

if __name__ == "__main__":
    data_ingestion.run()
    data_transformation.run()
    model_trainer.run()
    model_evaluation.run()
