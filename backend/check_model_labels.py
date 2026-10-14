
from transformers import AutoConfig

MODELS = {
    "Image": "dima806/deepfake_vs_real_image_detection",
    "Audio": "Vansh180/deepfake-audio-wav2vec2",
}


def main():
    for media_type, model_name in MODELS.items():
        print("\n" + "=" * 55)
        print(f"{media_type} model")
        print(f"Model ID: {model_name}")

        try:
            config = AutoConfig.from_pretrained(model_name)

            print("Model architecture:", config.architectures)
            print("Class labels:")

            for class_id, label in config.id2label.items():
                print(f"  {class_id}: {label}")

        except Exception as error:
            print(f"Could not inspect this model: {error}")


if __name__ == "__main__":
    main()
