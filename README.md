# TextSummarizer using Hugging Face

A machine learning project for automatic text summarization using Hugging Face Transformers.

## Features

- End-to-end training and prediction pipelines
- Modular configuration with YAML files
- Data ingestion, transformation, and model training components
- REST APIs for training and batch prediction

## Project Structure

- `config.yaml` – Main configuration file
- `params.yaml` – Model and training parameters
- `src/` – Source code for components and pipelines
- `api/` – API endpoints for training and prediction

## Workflows

1. **Configuration**: Define settings in `config.yaml` and `params.yaml`.
2. **Component Setup**: Implement data ingestion, transformation, and model training modules.
3. **Pipeline Creation**: Build training and prediction pipelines.
4. **API Integration**: Expose APIs for training and batch prediction.

## Getting Started

1. Clone the repository.
2. Install dependencies:  
    ```bash
    pip install -r requirements.txt
    ```
3. Update `config.yaml` and `params.yaml` as needed.
4. Run the training pipeline:  
    ```bash
    python src/train_pipeline.py
    ```
5. Start the API server:  
    ```bash
    uvicorn api.main:app --reload
    ```

## License

This project is licensed under the MIT License.