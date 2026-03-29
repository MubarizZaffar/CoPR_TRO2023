import hydra
from pipeline import SevenScenesRetrievalTest
HYDRA_FULL_ERROR=1
@hydra.main(config_path="configs", config_name="7scenes")
def main(cfg):
    benchmark = None
    if cfg.experiment_params.name == '7scenes':
        print(cfg)
        benchmark = SevenScenesRetrievalTest(cfg)

    benchmark.evaluate()


if __name__ == "__main__":
    main()
