# Syllabus Date Extractor

AI-powered tool for extracting important dates from syllabus PDFs using TensorFlow and pretrained NER models.

## Features

- **PDF Text Extraction**: Supports multiple PDF parsing methods (PyPDF2 and pdfplumber)
- **AI-Powered Date Detection**: Uses pretrained BERT-based NER models for intelligent date extraction
- **Custom Model Training**: Includes TensorFlow/Keras architecture for training custom models on syllabus-specific data
- **Date Classification**: Automatically classifies dates into categories (exams, assignments, holidays, etc.)
- **Multiple Extraction Methods**: Combines AI models with regex patterns for comprehensive date detection

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- TensorFlow (for custom model training)
- PyTorch and Transformers (for pretrained NER models)
- PyPDF2 and pdfplumber (for PDF processing)
- Other utilities (numpy, python-dateutil, etc.)

## Usage

### Basic Usage

```python
from syllabus_date_extractor import SyllabusDateProcessor

# Initialize the processor with pretrained models
processor = SyllabusDateProcessor(use_pretrained=True)

# Process a syllabus PDF
results = processor.process_syllabus('path/to/syllabus.pdf')

# Export results to JSON
processor.export_results(results, 'extracted_dates.json')

# Access the results
print(f"Found {results['num_dates']} dates")
for date in results['dates']:
    print(f"{date['text']} - Type: {date['type']}")
```

### Using Custom Model

```python
from syllabus_date_extractor import CustomDateModel

# Create and build custom model
custom_model = CustomDateModel(vocab_size=10000, embedding_dim=128)
model = custom_model.build_model()

# Train the model (with your prepared data)
history = custom_model.train(X_train, y_train, X_val, y_val, epochs=10)

# Save the trained model
custom_model.save_model('my_date_model')

# Load and use later
custom_model.load_model('my_date_model')
predictions = custom_model.predict(X_test)
```

### Components

#### 1. PDFTextExtractor
Extracts text from PDF files using multiple methods:
```python
from syllabus_date_extractor import PDFTextExtractor

extractor = PDFTextExtractor()
text = extractor.extract_text('syllabus.pdf', method='auto')
```

#### 2. DateExtractor
Extracts dates using pretrained models or regex:
```python
from syllabus_date_extractor import DateExtractor

date_extractor = DateExtractor(use_pretrained=True)
dates = date_extractor.extract_dates(text)
```

#### 3. CustomDateModel
Neural network for custom training:
```python
from syllabus_date_extractor import CustomDateModel

model = CustomDateModel()
model.build_model()
# Train with your data
```

## Model Architecture

The custom model uses:
- **Embedding Layer**: Converts text to dense vectors
- **Bidirectional LSTM**: Captures sequential patterns in both directions
- **Dense Layers**: Classification with dropout for regularization
- **Multi-label Output**: Classifies dates into 5 categories:
  - Exam dates
  - Assignment/homework dates
  - Holidays/breaks
  - Deadlines
  - Other important dates

## Output Format

Results are returned as a dictionary:
```json
{
  "status": "success",
  "text": "Full extracted text...",
  "dates": [
    {
      "text": "September 15, 2024",
      "start": 245,
      "end": 264,
      "type": "exam",
      "context": "...midterm exam on September 15, 2024...",
      "method": "pretrained_ner",
      "score": 0.95
    }
  ],
  "num_dates": 12
}
```

## Training Your Own Model

To train a custom model on your syllabus data:

1. **Prepare Training Data**: Create tokenized sequences and labels
2. **Build Model**: Use the CustomDateModel class
3. **Train**: Call the train() method with your data
4. **Evaluate**: Test on validation set
5. **Save**: Save the trained model for later use

```python
# Example training workflow
custom_model = CustomDateModel()
custom_model.build_model()

# Your training data (tokenized text and labels)
# X_train shape: (num_samples, max_length)
# y_train shape: (num_samples, 5) - one-hot encoded labels

history = custom_model.train(
    X_train, y_train,
    X_val, y_val,
    epochs=20,
    batch_size=32
)

custom_model.save_model('trained_model')
```

## Dependencies

- **tensorflow**: Deep learning framework for custom model
- **transformers**: Pretrained NER models (BERT-based)
- **PyPDF2**: PDF text extraction
- **pdfplumber**: Alternative PDF parser for complex layouts
- **torch**: Backend for transformers library
- **numpy**: Numerical operations

## Running the Demo

```bash
python syllabus_date_extractor.py
```

This will show:
- Available features and usage examples
- Model architecture summary
- Instructions for processing PDFs

## Project Structure

```
Syllabot/
├── syllabus_date_extractor.py  # Main script with all classes
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Future Enhancements

- Fine-tune pretrained models on syllabus-specific data
- Add more date classification categories
- Improve context analysis for better classification
- Support for more PDF formats and layouts
- Web interface for easy use
- Batch processing for multiple syllabi

## License

This project is part of Aggie Agenda.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.
