import hydra
from relposenet.pipeline import Pipeline

"""
Task for Mubariz
1. Add negative sampling in data loader
2. Add feature descriptor computation for negative samples
3. Add DB feature descs storage
4. Add faiss-based feature matching
5. Add Recall-rate measurement
6. Add VLAD Module and do multi-task learning

"""
@hydra.main(config_path="configs", config_name="main")
def main(cfg):
    pipeline = Pipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
