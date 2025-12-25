"""
Syllabus Date Extractor
A Python script for extracting text from PDF syllabi and finding important dates
using AI models (both pretrained and custom trainable models).
"""

import tensorflow as tf
import numpy as np
import re
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import json

# PDF Processing
try:
    import PyPDF2
except ImportError:
    print("Warning: PyPDF2 not installed. Install with: pip install PyPDF2")

# Alternative PDF processing library
try:
    import pdfplumber
except ImportError:
    print("Warning: pdfplumber not installed. Install with: pip install pdfplumber")

# Natural Language Processing
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
except ImportError:
    print("Warning: transformers not installed. Install with: pip install transformers")

# For custom model training
try:
    from tensorflow import keras
    from tensorflow.keras import layers
except ImportError:
    print("Warning: Keras not properly installed")


class PDFTextExtractor:
    """Extract text from PDF files using multiple methods."""
    
    def __init__(self):
        self.text_content = ""
        
    def extract_with_pypdf2(self, pdf_path: str) -> str:
        """
        Extract text from PDF using PyPDF2.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as string
        """
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text()
                    
            return text
        except Exception as e:
            print(f"Error extracting with PyPDF2: {e}")
            return ""
    
    def extract_with_pdfplumber(self, pdf_path: str) -> str:
        """
        Extract text from PDF using pdfplumber (better for complex layouts).
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as string
        """
        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                        
            return text
        except Exception as e:
            print(f"Error extracting with pdfplumber: {e}")
            return ""
    
    def extract_text(self, pdf_path: str, method: str = "auto") -> str:
        """
        Extract text from PDF using the specified or best available method.
        
        Args:
            pdf_path: Path to the PDF file
            method: Extraction method ("pypdf2", "pdfplumber", or "auto")
            
        Returns:
            Extracted text as string
        """
        if method == "pypdf2":
            self.text_content = self.extract_with_pypdf2(pdf_path)
        elif method == "pdfplumber":
            self.text_content = self.extract_with_pdfplumber(pdf_path)
        else:  # auto
            # Try pdfplumber first as it handles complex layouts better
            self.text_content = self.extract_with_pdfplumber(pdf_path)
            if not self.text_content:
                self.text_content = self.extract_with_pypdf2(pdf_path)
                
        return self.text_content


class DateExtractor:
    """Extract and classify important dates from text using AI models."""
    
    def __init__(self, use_pretrained: bool = True):
        """
        Initialize the date extractor.
        
        Args:
            use_pretrained: Whether to use pretrained models (default: True)
        """
        self.use_pretrained = use_pretrained
        self.pretrained_ner = None
        self.custom_model = None
        
        if use_pretrained:
            self._load_pretrained_model()
    
    def _load_pretrained_model(self):
        """Load pretrained NER model for date extraction."""
        try:
            # Using a pretrained NER model from HuggingFace
            # This model can identify dates and other named entities
            self.pretrained_ner = pipeline(
                "ner",
                model="dslim/bert-base-NER",
                aggregation_strategy="simple"
            )
            print("Pretrained NER model loaded successfully")
        except Exception as e:
            print(f"Error loading pretrained model: {e}")
            self.pretrained_ner = None
    
    def extract_dates_regex(self, text: str) -> List[Dict[str, str]]:
        """
        Extract dates using regex patterns as a fallback method.
        
        Args:
            text: Input text to extract dates from
            
        Returns:
            List of dictionaries with date information
        """
        date_patterns = [
            # MM/DD/YYYY or MM-DD-YYYY
            r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',
            # Month DD, YYYY
            r'\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b',
            # Month DD
            r'\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2})\b',
            # DD Month YYYY
            r'\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b',
        ]
        
        dates_found = []
        for pattern in date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                dates_found.append({
                    'text': match.group(1),
                    'start': match.start(),
                    'end': match.end(),
                    'method': 'regex'
                })
        
        return dates_found
    
    def extract_dates_pretrained(self, text: str) -> List[Dict[str, str]]:
        """
        Extract dates using pretrained NER model.
        
        Args:
            text: Input text to extract dates from
            
        Returns:
            List of dictionaries with date information
        """
        if not self.pretrained_ner:
            print("Pretrained model not available, using regex fallback")
            return self.extract_dates_regex(text)
        
        try:
            # Split text into chunks if it's too long (model has token limits)
            max_length = 512
            chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
            
            all_dates = []
            offset = 0
            
            for chunk in chunks:
                entities = self.pretrained_ner(chunk)
                
                # Filter for date-related entities
                for entity in entities:
                    # Check if entity is likely a date
                    if self._is_date_entity(entity):
                        all_dates.append({
                            'text': entity['word'],
                            'start': entity['start'] + offset,
                            'end': entity['end'] + offset,
                            'score': entity['score'],
                            'method': 'pretrained_ner'
                        })
                
                offset += len(chunk)
            
            # Combine with regex for better coverage
            regex_dates = self.extract_dates_regex(text)
            all_dates.extend(regex_dates)
            
            # Remove duplicates
            unique_dates = self._deduplicate_dates(all_dates)
            
            return unique_dates
            
        except Exception as e:
            print(f"Error with pretrained model: {e}")
            return self.extract_dates_regex(text)
    
    def _is_date_entity(self, entity: Dict) -> bool:
        """
        Check if an entity is likely to be a date.
        
        Args:
            entity: Entity dictionary from NER model
            
        Returns:
            True if entity is likely a date
        """
        # Check entity label and text for date patterns
        date_keywords = ['DATE', 'TIME', 'CARDINAL']
        text = entity.get('word', '')
        
        # Check if text matches date patterns
        has_date_pattern = bool(re.search(r'\d+[/-]\d+|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', text, re.IGNORECASE))
        
        return has_date_pattern or entity.get('entity_group') in date_keywords
    
    def _deduplicate_dates(self, dates: List[Dict]) -> List[Dict]:
        """
        Remove duplicate dates that overlap in position.
        
        Args:
            dates: List of date dictionaries
            
        Returns:
            Deduplicated list of dates
        """
        if not dates:
            return []
        
        # Sort by start position
        sorted_dates = sorted(dates, key=lambda x: x['start'])
        
        unique_dates = [sorted_dates[0]]
        
        for date in sorted_dates[1:]:
            # Check if this date overlaps with the last unique date
            last_date = unique_dates[-1]
            if date['start'] >= last_date['end']:
                unique_dates.append(date)
            elif date.get('score', 0) > last_date.get('score', 0):
                # Replace if this date has higher confidence
                unique_dates[-1] = date
        
        return unique_dates
    
    def extract_dates(self, text: str) -> List[Dict[str, str]]:
        """
        Extract dates from text using configured method.
        
        Args:
            text: Input text to extract dates from
            
        Returns:
            List of dictionaries with date information
        """
        if self.use_pretrained and self.pretrained_ner:
            return self.extract_dates_pretrained(text)
        else:
            return self.extract_dates_regex(text)


class CustomDateModel:
    """
    Custom TensorFlow model for date extraction and classification.
    This is a template for training your own model on syllabus-specific data.
    """
    
    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 128, max_length: int = 100):
        """
        Initialize the custom model architecture.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings
            max_length: Maximum sequence length
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
        
    def build_model(self) -> tf.keras.Model:
        """
        Build a neural network model for date detection and classification.
        
        Returns:
            Compiled Keras model
        """
        # Input layer
        inputs = layers.Input(shape=(self.max_length,))
        
        # Embedding layer
        x = layers.Embedding(self.vocab_size, self.embedding_dim)(inputs)
        
        # Bidirectional LSTM layers
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Bidirectional(layers.LSTM(32))(x)
        x = layers.Dropout(0.3)(x)
        
        # Dense layers
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # Output layer for multi-label classification
        # Labels: exam_date, assignment_date, holiday, deadline, other_date
        outputs = layers.Dense(5, activation='sigmoid')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        self.model = model
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 10, batch_size: int = 32):
        """
        Train the custom model.
        
        Args:
            X_train: Training data (tokenized sequences)
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
        
        # Early stopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
        
        # Model checkpoint callback
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            'best_date_model.h5',
            monitor='val_loss',
            save_best_only=True
        )
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, checkpoint]
        )
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input data (tokenized sequences)
            
        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model not built or loaded")
        
        return self.model.predict(X)
    
    def save_model(self, path: str = "date_extraction_model"):
        """
        Save the trained model.
        
        Args:
            path: Path to save the model
        """
        if self.model:
            self.model.save(path)
            print(f"Model saved to {path}")
    
    def load_model(self, path: str = "date_extraction_model"):
        """
        Load a trained model.
        
        Args:
            path: Path to the saved model
        """
        self.model = tf.keras.models.load_model(path)
        print(f"Model loaded from {path}")


class SyllabusDateProcessor:
    """
    Main class that orchestrates PDF reading and date extraction.
    """
    
    def __init__(self, use_pretrained: bool = True, use_custom_model: bool = False):
        """
        Initialize the syllabus processor.
        
        Args:
            use_pretrained: Whether to use pretrained models
            use_custom_model: Whether to use custom trained model
        """
        self.pdf_extractor = PDFTextExtractor()
        self.date_extractor = DateExtractor(use_pretrained=use_pretrained)
        self.custom_model = None
        
        if use_custom_model:
            self.custom_model = CustomDateModel()
    
    def process_syllabus(self, pdf_path: str) -> Dict:
        """
        Process a syllabus PDF and extract important dates.
        
        Args:
            pdf_path: Path to the syllabus PDF file
            
        Returns:
            Dictionary with extracted text and dates
        """
        print(f"Processing syllabus: {pdf_path}")
        
        # Extract text from PDF
        print("Extracting text from PDF...")
        text = self.pdf_extractor.extract_text(pdf_path)
        
        if not text:
            return {
                'status': 'error',
                'message': 'Could not extract text from PDF',
                'text': '',
                'dates': []
            }
        
        print(f"Extracted {len(text)} characters")
        
        # Extract dates
        print("Extracting dates...")
        dates = self.date_extractor.extract_dates(text)
        
        print(f"Found {len(dates)} dates")
        
        # Classify dates by context (simple rule-based for now)
        classified_dates = self._classify_dates(dates, text)
        
        return {
            'status': 'success',
            'text': text,
            'dates': classified_dates,
            'num_dates': len(dates)
        }
    
    def _classify_dates(self, dates: List[Dict], context: str) -> List[Dict]:
        """
        Classify dates by type based on surrounding context.
        
        Args:
            dates: List of extracted dates
            context: Full text for context analysis
            
        Returns:
            List of classified dates
        """
        classified = []
        
        # Keywords for different date types
        exam_keywords = ['exam', 'test', 'midterm', 'final', 'quiz']
        assignment_keywords = ['assignment', 'homework', 'project', 'due', 'submit']
        holiday_keywords = ['holiday', 'break', 'recess', 'no class']
        
        for date in dates:
            start = max(0, date['start'] - 50)
            end = min(len(context), date['end'] + 50)
            surrounding_text = context[start:end].lower()
            
            date_type = 'other'
            
            if any(keyword in surrounding_text for keyword in exam_keywords):
                date_type = 'exam'
            elif any(keyword in surrounding_text for keyword in assignment_keywords):
                date_type = 'assignment'
            elif any(keyword in surrounding_text for keyword in holiday_keywords):
                date_type = 'holiday'
            
            classified.append({
                **date,
                'type': date_type,
                'context': surrounding_text.strip()
            })
        
        return classified
    
    def export_results(self, results: Dict, output_path: str):
        """
        Export results to a JSON file.
        
        Args:
            results: Processing results
            output_path: Path to save JSON file
        """
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results exported to {output_path}")


def main():
    """
    Main function demonstrating usage of the syllabus date extractor.
    """
    print("Syllabus Date Extractor")
    print("=" * 50)
    print("\nThis script extracts dates from syllabus PDFs using:")
    print("- TensorFlow for custom model training")
    print("- Pretrained NER models for date detection")
    print("- PDF processing libraries for text extraction")
    print("\n" + "=" * 50)
    
    # Example usage
    processor = SyllabusDateProcessor(use_pretrained=True)
    
    # If you have a PDF file, process it:
    # results = processor.process_syllabus('path/to/syllabus.pdf')
    # processor.export_results(results, 'extracted_dates.json')
    
    print("\nTo use this script:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Create a SyllabusDateProcessor instance")
    print("3. Call process_syllabus() with your PDF path")
    print("\nExample:")
    print("  processor = SyllabusDateProcessor(use_pretrained=True)")
    print("  results = processor.process_syllabus('syllabus.pdf')")
    print("  processor.export_results(results, 'dates.json')")
    
    # Example: Building custom model architecture
    print("\n" + "=" * 50)
    print("Custom Model for Training:")
    print("=" * 50)
    custom_model = CustomDateModel()
    model = custom_model.build_model()
    print("\nModel Summary:")
    model.summary()
    
    print("\nTo train the custom model:")
    print("  custom_model = CustomDateModel()")
    print("  custom_model.build_model()")
    print("  history = custom_model.train(X_train, y_train, X_val, y_val)")
    print("  custom_model.save_model('my_date_model')")


if __name__ == "__main__":
    main()
