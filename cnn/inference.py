from utils.inference import InferenceConfig, run_inference

def main(argv: list[str] | None = None) -> None:
    run_inference(InferenceConfig.from_args(argv))

if __name__ == '__main__':
    main()
