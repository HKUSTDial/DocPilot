from typing import Dict, List, Tuple, Type, Optional, Union, Any

from pydantic import BaseModel, Field, create_model
import re

from abc import ABC, abstractmethod

import json
import xml.etree.ElementTree as ET

from .utils import sanitize


def try_parse_nested(value):
    # 尝试解析为 JSON
    try:
        return json.loads(value)
    except Exception:
        pass
    # 尝试解析为 XML
    try:
        element = ET.fromstring(value)
        return {child.tag: try_parse_nested(child.text or "") for child in element}
    except Exception:
        pass
    # 都不是，原样返回
    return value

class FormatError(Exception):
    """Exception raised when response format validation fails"""
    pass

class BaseFormatter(BaseModel):
    """Base class for all formatters"""
    
    @abstractmethod
    def prepare_prompt(self, prompt: str) -> str:
        """Prepare the prompt to instruct the LLM to return in the required format"""
        pass
    
    @abstractmethod
    def validate_response(self, response: str) -> Tuple[bool, Any]:
        """Validate if the response matches the expected format"""
        pass

    def format_error_message(self) -> str:
        """Return an error message for invalid format"""
        return f"Response did not match the expected {self.__class__.__name__} format"

class XmlFormatter(BaseFormatter):
    """Formatter for XML responses"""
    model: Optional[Type[BaseModel]] = None
    fields: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_dict(cls, fields_dict: Dict[str, str]) -> "XmlFormatter":
        """
        Create formatter from a dictionary of field names and descriptions
        
        Args:
            fields_dict: Dictionary where keys are field names and values are field descriptions
            
        Returns:
            An XmlFormatter instance configured with the specified fields
        """
        model_fields = {}
        for name, desc in fields_dict.items():
            model_fields[name] = (str, Field(default="", description=desc))
        
        model_class = create_model("XmlResponseModel", **model_fields)
        
        return cls(model=model_class)
    
    @classmethod
    def from_model(cls, model_class: Type[BaseModel]) -> "XmlFormatter":
        """
        Create formatter from an existing Pydantic model class
        
        Args:
            model_class: A Pydantic model class
            
        Returns:
            An XmlFormatter instance configured with the model's fields
        """
        return cls(model=model_class)
    
    def _get_field_names(self) -> List[str]:
        """Get field names from the model"""
        if self.model:
            return list(self.model.model_fields.keys())
        return []
    
    def _get_field_description(self, field_name: str) -> str:
        """Get field description from the model"""
        if self.model and field_name in self.model.model_fields:
            return self.model.model_fields[field_name].description
        return ""    
    
    def prepare_prompt(self, prompt: str) -> str:
        examples = []
        for field_name in self._get_field_names():
            description = self._get_field_description(field_name)
            examples.append(f"<{field_name}>{description}</{field_name}>")

        example_str = "\n".join(examples)
        
        instructions = prompt + f"\n# You can output anything, but in the final, you need to provide the following fields, and make sure to warp it with xml tags:\n{example_str}"
        return instructions
    
    def validate_response(self, response: str) -> Tuple[bool, dict]:
        """Validate if the response contains all required fields in XML format"""
        try:
            pattern = r"<(\w+)>(.*?)</\1>"
            matches = re.findall(pattern, response, re.DOTALL)
            found_fields = {match[0]: try_parse_nested(match[1].strip()) for match in matches}
            
            # found_fields = {match[0]: match[1].strip() for match in matches}
            
            for field_name in self._get_field_names():
                field = self.model.model_fields[field_name]
                is_required = field.default is None and field.default_factory is None
                
                if is_required and (field_name not in found_fields or not found_fields[field_name]):
                    raise FormatError(f"Field '{field_name}' is missing or empty.")

            return True, found_fields
        except Exception:
            return False, None
 
class CodeFormatter(BaseFormatter):
    """
    Formatter for extracting and sanitizing code from LLM responses.
    Handles both markdown code blocks and raw code responses.
    """
    
    function_name: Optional[str] = None
    
    def prepare_prompt(self, prompt: str) -> str:
        """
        Prepare the prompt to instruct the LLM to return code in a proper format.
        
        Args:
            prompt: The original prompt
            
        Returns:
            The prompt with instructions to return code in markdown format
        """
        # Instructions to return code in appropriate format
        code_instructions = (
            "\n\n"
            "Please write your code solution in Python. "
            "Return ONLY the complete, runnable code without explanations. "
            "Use proper Python syntax and formatting. "
        )
        
        # Add function-specific instructions if function_name is provided
        if self.function_name:
            code_instructions += (
                f"\nMake sure to include a function named '{self.function_name}' in your solution. "
                f"This function will be the entry point for the program."
            )
        
        return prompt + code_instructions
    
    def validate_response(self, response: str) -> Tuple[bool, Union[Dict[str, str], str, None]]:
        """
        Extract code from response and validate it.
        
        Args:
            response: The LLM response
            
        Returns:
            A tuple with (is_valid, extracted_code)
        """
        try:
            # First try to extract code from markdown code blocks
            code = self._extract_code_from_markdown(response)
    
            # If no code blocks found, treat the entire response as code
            if not code:
                code = response
            
            # Use the sanitize function to extract valid code and handle dependencies
            sanitized_code = sanitize(code=code, entrypoint=self.function_name)
            
            # If sanitize returned empty string, the code is invalid
            if not sanitized_code.strip():
                return False, {"error": "No valid code could be extracted"}
            
            # Check if required function exists
            if self.function_name and self.function_name not in sanitized_code:
                return False, {"error": f"Required function '{self.function_name}' not found in code"}
            
            # Return the sanitized code
            result = {"response": sanitized_code}
            return True, result
            
        except Exception as e:
            # Return the error information with more context
            error_type = type(e).__name__
            return False, {"error": f"{error_type}: {str(e)}"}
    
    def _extract_code_from_markdown(self, text: str) -> str:
        """
        Extract code from markdown code blocks in the response.
        
        Args:
            text: The text containing possible markdown code blocks
            
        Returns:
            The extracted code as a string, or empty string if no code blocks found
        """
        # Look for Python code blocks with various delimiters
        patterns = [
            r"```python\s*([\s\S]*?)\s*```",  # ```python ... ```
            r"~~~python\s*([\s\S]*?)\s*~~~",  # ~~~python ... ~~~
            r"```\s*([\s\S]*?)\s*```",        # ``` ... ```
            r"~~~\s*([\s\S]*?)\s*~~~",        # ~~~ ... ~~~
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                # Join all code blocks
                return "\n\n".join(matches)
        
        # No code blocks found
        return ""
    
    def format_error_message(self) -> str:
        """Return a helpful error message if code validation fails"""
        base_message = "Could not extract valid Python code from the response."
        if self.function_name:
            return f"{base_message} Make sure the code includes a function named '{self.function_name}'."
        return base_message

    @classmethod
    def create(cls, function_name: Optional[str] = None) -> "CodeFormatter":
        """
        Factory method to create a CodeFormatter instance
        
        Args:
            function_name: Optional name of the function to extract
            
        Returns:
            A configured CodeFormatter instance
        """
        return cls(function_name=function_name)        
        


class TextFormatter(BaseFormatter):    
    def prepare_prompt(self, prompt: str) -> str:
        return prompt
    
    def validate_response(self, response: str) -> Tuple[bool, Union[str, None]]:
        """
        For plain text formatter, we simply return the response as is without validation
        since there are no format restrictions
        """
        return True, response
    
