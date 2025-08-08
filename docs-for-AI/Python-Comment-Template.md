Hello fellow AI\! I'm here to guide you on how to effectively apply a comprehensive Python comment template to existing Python code. This template is designed to enhance code readability, maintainability, and understandability for both humans and AI systems.

The core idea is to systematically go through each Python file and integrate the provided comment structure. This involves adding module-level documentation, class-level documentation, and function/method-level documentation, along with other structural comments.

Let's break down the implementation process step-by-step:

### 1\. **Understanding the Template Structure**

First, familiarize yourself with the different sections of the template:

  * **File Header:** Provides metadata about the file, its project context, author, dates, and file path.
  * **Module Objective:** A high-level summary of the file's overall purpose.
  * **Module Contents Overview:** A bulleted list of the main components (classes, functions, constants) within the file.
  * **Dependencies / Imports:** Lists external and internal modules that the current file relies on.
  * **Usage / Integration:** Explains how this module is intended to be used by other parts of the project.
  * **Imports Section:** Structured imports for standard, third-party, and local libraries.
  * **Global Constants / Variables:** Defines module-level constants.
  * **Class Definitions:** Detailed documentation for each class, including its purpose, attributes, and methods.
      * **Class Variables:** Section for class-level variables.
      * **Constructor (`__init__`)**: Detailed documentation for the class constructor.
      * **Destructor (`__del__`)**: (Use sparingly) Documentation for cleanup operations.
      * **Getters (Properties):** Documentation for property-based getters and traditional getter methods.
      * **Setters / Mutators:** Documentation for methods that modify class state.
      * **Internal/Private Methods:** Documentation for methods intended for internal use.
      * **Class Information Methods (`__str__`, `__repr__`):** Documentation for string representations.
  * **Standalone Function Definitions:** Detailed documentation for functions that are not part of a class.
      * **Utility Functions:** General-purpose helper functions.
      * **Helper Functions:** Functions specific to the module's internal operations.
      * **Callable Functions from other modules:** Functions designed to be exposed and called by other modules.
  * **Module Initialization / Main Execution Guard (`if __name__ == '__main__':`)**: Contains code that runs only when the script is executed directly, typically for testing or examples.

### 2\. **Step-by-Step Implementation Guide**

**For each Python file in your project:**

#### **Phase 1: Top-Level Module Comments**

1.  **Start with the File Header:** Copy and paste the entire "File Header" block at the very top of your Python file.

      * Fill in `[your_module_name.py]` with the actual filename (e.g., `data_validators.py`).
      * Ensure the `Author`, `Date`, and `Last Modified` fields are accurate.
      * Update `[File Path]` to reflect the module's location within the project.

2.  **Add Module Objective:** After the file header, copy and paste the "Module Objective" block.

      * Write a concise summary of what the *specific file* does. What is its main responsibility? What problem does it solve?

3.  **Populate Module Contents Overview:** Copy and paste the "Module Contents Overview" block.

      * Go through your file and list all major classes, functions, and significant constants defined within it. This acts as a quick reference.

4.  **List Dependencies / Imports:** Copy and paste the "Dependencies / Imports" block.

      * **Standard Library:** Identify and list any modules imported from Python's standard library (e.g., `os`, `sys`, `json`, `datetime`).
      * **Third-Party:** List any external libraries installed via pip (e.g., `pandas`, `requests`, `numpy`, `flask`).
      * **Local Project Modules:** List imports from other files within your project, providing relative or absolute paths as appropriate (e.g., `from .config import settings`).

5.  **Explain Usage / Integration:** Copy and paste the "Usage / Integration" block.

      * Describe how this module is meant to be used. Which other modules will import from it? What is its role in the overall application flow?

#### **Phase 2: Organizing Imports and Global Constants**

1.  **Imports Section:** Locate your existing `import` statements.

      * Cut and paste them into the "Imports" section of the template.
      * Organize them into "Standard library imports," "Third-party library imports," and "Local application/library specific imports."
      * Ensure each import has a brief, inline comment explaining its purpose if it's not immediately obvious.

2.  **Global Constants / Variables:** Identify any global constants in your file (variables defined at the module level, usually in `ALL_CAPS`).

      * Move them under the "Global Constants / Variables" header.
      * Add a concise comment for each constant explaining its purpose.

#### **Phase 3: Documenting Classes**

1.  **For each `class` definition:**
      * Copy the "Class Definitions" template block and place it immediately above your class.

      * **Class Docstring (`"""..."""`):**

          * Fill in the concise one-line summary.
          * Provide a more detailed explanation of the class's responsibilities, its role, and how it interacts with other components.
          * List `Class Attributes`, `Instance Attributes`, and `Methods` using the provided format. Ensure you document private attributes (with `__` prefix) and internal methods (with `_` prefix).

      * **Class Variables:** Place any class-level variables under the "--- Class Variable ---" section.

      * **Constructor (`__init__`)**:

          * Ensure your `__init__` method has a docstring immediately following its `def` line.
          * Fill in the summary and detailed explanation.
          * Document all `Args:` (parameters) and their types and descriptions.

      * **Destructor (`__del__`)**: (If your class has one, which is rare for general Python applications)

          * Place the `__del__` method and its docstring under the "--- Destructor ---" section.
          * Explain what resources it cleans up.

      * **Getters / Properties (`@property`):**

          * For each getter method or property, ensure it has a docstring.
          * Place them under the "--- Getters ---" section.

      * **Setters / Mutators:**

          * For each method that modifies the object's state, ensure it has a docstring.
          * Place them under the "--- Setters / Mutators ---" section.

      * **Internal/Private Methods (`_method_name`):**

          * For methods prefixed with a single underscore, ensure they have docstrings.
          * Explain their internal purpose and note that they are not intended for direct external use.
          * Place them under the "--- Internal/Private Methods ---" section.

      * **Class Information Methods (`__str__`, `__repr__`):**

          * If your class implements `__str__` or `__repr__`, add the provided docstrings.
          * Explain the purpose of each: `__str__` for user-friendly display, `__repr__` for unambiguous developer representation.

      * **Wrap your class methods between two comment lines using "---", like so:**

        ```python
        # --------------------------------------------------------------------------------- my_method ()
        def my_method(self):
            # ...
            pass
        # --------------------------------------------------------------------------------- enf my_method()
        ```

        And for the class variable section:

        ```python
        # ----------------------
        # --- Class Variable ---
        # ----------------------
        _instance_count = 0
        # ---------------------------------------------------------------------------------
        ```

#### **Phase 4: Documenting Standalone Functions**

1.  **For each standalone `def` function:**
      * Copy the "Standalone Function Definitions" template block and place it immediately above your function.

      * **Function Docstring (`"""..."""`):**

          * Fill in the concise one-line summary.
          * Provide a more detailed explanation of what the function does, its algorithm, and any side effects.
          * Document all `Args:` (parameters), their types, and descriptions.
          * Document the `Returns:` value and its type.
          * If applicable, document `Raises:` exceptions.
          * Provide `Examples:` using `>>>` for doctests if appropriate.

      * **Categorize Functions:** Place functions under the appropriate sub-headers: "Utility Functions," "Helper Functions," or "Callable Functions from other modules."

      * **Wrap your functions between two comment lines using "---", like so:**

        ```python
        # --------------------------------------------------------------------------------- my_function()
        def my_function(param):
            # ...
            pass
        # --------------------------------------------------------------------------------- end my_function()
        ```

#### **Phase 5: Module Initialization / Main Execution Guard**

1.  **`if __name__ == '__main__':` block:**
      * If your file has a `if __name__ == '__main__':` block, ensure it is clearly commented.
      * Explain its purpose: typically for module-specific tests or example usage when the file is run directly.
      * Add comments within this block to describe any test cases or example code.
      * **Wrap this block with the "---" comments as well.**

### 3\. **Important Considerations for AI Implementation**

  * **Parsers and Linters:** If you are an AI that will be automating this process, consider using Abstract Syntax Tree (AST) parsers (like Python's built-in `ast` module) to programmatically identify classes, functions, and their parameters. This will allow you to insert comments accurately.
  * **Idempotency:** Your implementation should be idempotent. Running the comment application process multiple times on the same file should not lead to duplicate comments or corrupted structures. You'll need logic to detect existing comments and update them, rather than simply adding new ones.
  * **Existing Docstrings:** If existing code already has docstrings, your system should merge or intelligently replace them, rather than overwriting valuable existing information. A good approach would be to:
    1.  Extract the existing docstring content.
    2.  Parse its sections (Args, Returns, etc.).
    3.  Integrate this information into the new template's structured sections.
    4.  Prompt for human review if conflicts or ambiguities arise.
  * **Type Hinting:** The template uses type hinting (e.g., `param: [type]`). When applying to existing code, if type hints are missing, your AI might suggest adding them or make an educated guess based on variable usage (though this is more advanced).
  * **Contextual Understanding:** For the "Module Objective," "Usage / Integration," and detailed explanations in docstrings, a deeper understanding of the code's purpose and its role within the larger project is crucial. This might require analyzing call graphs, data flows, or even project-level documentation.
  * **Iterative Refinement:** It's unlikely a perfect set of comments will be generated on the first pass. Implement a feedback loop or a review process (human or automated) to refine the generated comments.

By following this structured approach, you can systematically apply the comment template to Python code, significantly improving its documentation and making it more accessible for future development and maintenance.

## Make sure that you implement the class and function divided correctly and end of file header
for example:
```Python
# ------------------------------------------------------------------------- MyClassName
Class body
# ------------------------------------------------------------------------- end MyClassName

# ------------------------------------------------------------------------- my_function_name()
Function body
# ------------------------------------------------------------------------- end my_function_name()

# =========================================================================
# End of File
# =========================================================================
```


### ⚠️ BECAREFUL! WHEN ADDING COMMENTS DO NOT MODIFY THE EXISTING CODE JUST ADD THE COMMENTS USING THE DIRECTION ABOVE

#### Template:

```python

# -------------------------------------------------------------------------
# File: [your_module_name.py] (e.g., models.py, data_validators.py, ui_handlers.py)
# Project: APH-IF
# Author: Alexander Ricciardi
# Date: 08-03-2025
# [File Path] (e.g., backend/tools/your_module_name.py)
# ------------------------------------------------------------------------

# --- Module Objective ---
#   [A concise, high-level summary of what this *specific file/module* does.]
#   [Explain its purpose, the main functionalities it provides, and its role
#   within the larger project or package.]
#   Example: "This module defines the `HomeInventory` class, managing data
#   persistence and core CRUD operations for homes via file I/O."
#   Example: "Contains utility functions for validating various types of user
#   input, ensuring data integrity across the application."
# -------------------------------------------------------------------------

# --- Module Contents Overview ---
# [A brief, bulleted list of the primary classes, functions, or logical
#  components defined within this file. This helps at a glance.]
# - Class: [MyClassName]
# - Function: [my_utility_function]
# - Function: [another_related_function]
# - Constants: [MY_GLOBAL_CONSTANT]
# -------------------------------------------------------------------------

# --- Dependencies / Imports ---
# [List any crucial external (third-party) or internal (local) modules
#  this specific file relies on. This helps understand module coupling.]
# - Standard Library: os (for file operations), sys (for path manipulation)
# - Third-Party: pandas (if using dataframes), requests (if making HTTP calls)
# - Local Project Modules:
#   - from .config import settings (if 'config' is a sibling module)
#   - from my_package.database import DBManager (if part of a larger package)
# -------------------------------------------------------------------------

# --- Usage / Integration ---
# [Explain how this file/module is intended to be used or integrated
#  by other parts of the project. Who should import it, and why?]
# Example: "The `HomeInventory` class from this module should be instantiated
#   by `main.py` or `app.py` to manage home data."
# Example: "Functions from this module (e.g., `validate_email`) are imported
#   into `user_management.py` and `forms.py` for input validation."

# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------

"""

Description of the module functionality

example:
Circuit Breaker Implementation for External Service Resilience

Provides circuit breaker pattern to protect against cascading failures
when external services (Neo4j, OpenAI, Gemini) become unavailable.

"""

# =========================================================================
# Imports
# =========================================================================
# Standard library imports
# import os
# import sys

# Third-party library imports
# import pandas as pd

# Local application/library specific imports
# from .utils import some_helper_function
# from ..config import APP_SETTINGS # Example for relative import in packages


# =========================================================================
# Global Constants / Variables
# =========================================================================
# Use ALL_CAPS for constants relevant to this module.
# MAX_RETRIES = 5 # Maximum attempts for an operation in this module
# DEFAULT_CHUNK_SIZE = 1024 # Buffer size for file operations


# =========================================================================
# Class Definitions
# =========================================================================

# ------------------------------------------------------------------------- class MyClassName
class [MyClassName]:
    """[A concise one-line summary of the class's purpose within this module.]

    [Provide a more detailed explanation of the class's responsibilities.]
    [Describe its key attributes, how it interacts with other components, and its role
    within the overall program. Discuss any invariants or pre/post conditions.]

    Class Attributes:
        [__class_attr_name (type)]: [Description of the class-level attribute.]
                                  [e.g., __instance_count (int): Tracks objects of this type.]

    Instance Attributes:
        [__instance_attr_name (type)]: [Description of the private instance attribute.]
        [public_attr_name (type)]: [Description of the public instance attribute.]

    Methods:
        [method_name()]: [Brief summary of each public method.]
        [_private_method()]: [Brief summary of each internal/private method.]
    """
    # ----------------------
    # --- Class Variable ---
    # ----------------------
    # [Define class-level variables here, if any.]
    _instance_count = 0
    
    # ---------------------------------------------------------------------------------

    # -------------------
    # --- Constructor ---
    # -------------------
    
    # --------------------------------------------------------------------------------- __init__()
    def __init__(self, [param1]: [type], [param2]: [type] = [default_value]) -> None:
        """Initializes the [MyClassName] object.

        [Provide a detailed explanation of what the constructor does.]
        [Describe any setup, initial state, or validations performed during object creation.]

        Args:
            [param1] ([type]): [Description of param1.]
            [param2] ([type], optional): [Description of param2.] Defaults to [default_value].
        """
        # [Initialize instance attributes here]
        self.__private_data = []
        self.public_property = [param1]
        # self.__class__._instance_count += 1  
    # --------------------------------------------------------------------------------- __init__()

    # ---------------------------------------------------------------------------------
    # --- Destructor (Use only if absolutely necessary for external resource cleanup) -
    # ---------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------- end __del__()
    def __del__(self) -> None:
        """Performs cleanup operations when the object is destroyed.
    
        [Explain what resources are being released or actions performed.]
        [e.g., Closes file handles, disconnects from a database, cleans up temp files.]
        """
        # [Cleanup code here]
        # print(f"Object {self.public_property} is being destroyed.")
    # --------------------------------------------------------------------------------- end __del__()

    # -----------------------
    # -- Embedded Classes --
    # ----------------------

    # --------------------------------------------------------------------------------- MyEmbeddedClass
    class [MyEmbeddedClass]
        # embedded class body logic
    # --------------------------------------------------------------------------------- end MyEmbeddedClass

    # -----------------------------------------------------------------------------
    # --- Getters (Property decorators are often preferred for simple getters) ---
    # -----------------------------------------------------------------------------

    # --------------------------------------------------------------------------------- some_property()
    @property
    def some_property(self) -> [type]:
        """[Retrieves the value of some_property.]"""
        return self.__private_data
    # --------------------------------------------------------------------------------- end some_property()

    # --------------------------------------------------------------------------------- get_data_by_id()
    def get_data_by_id(self, item_id: int) -> [type]:
        """Retrieves data based on a unique identifier.
    
        Args:
            item_id (int): The ID of the item to retrieve.
    
        Returns:
            [type]: The retrieved data, or None if not found.
        """
        # [Logic to retrieve data]
        pass
    # --------------------------------------------------------------------------------- end get_data_by_id()

    # ---------------------------
    # --- Setters / Mutators ---
    # ---------------------------
    
    # --------------------------------------------------------------------------------- set_value()
    def set_value(self, new_value: [type]) -> None:
        """Sets a new value for a specific attribute.
    
        Args:
            new_value ([type]): The value to set.
        """
        # [Logic to set/update data]
        pass
    # --------------------------------------------------------------------------------- end set_value()

    # --------------------------------------------------------------------------------- add_item()
    def add_item(self, item_data: dict) -> bool:
        """Adds a new item to the collection.
    
        Args:
            item_data (dict): Dictionary containing the item's details.
    
        Returns:
            bool: True if item was added successfully, False otherwise.
        """
        # [Logic to add item]
        pass
    # --------------------------------------------------------------------------------- end add_item()

    # -----------------------------------------------------------------------
    # --- Internal/Private Methods (Single leading underscore convention) ---
    # -----------------------------------------------------------------------

    # --------------------------------------------------------------------------------- _process_internal_data()
    def _process_internal_data(self, raw_data: list) -> list:
        """An internal helper method to process raw data.
    
        This method is not intended for direct external use.
    
        Args:
            raw_data (list): The raw data to be processed.
    
        Returns:
            list: The processed data.
        """
        # [Internal processing logic]
        return [processed_data]
    # --------------------------------------------------------------------------------- end _process_internal_data()

    # ---------------------------------------------------------------------
    # --- Class Information Methods (Optional, but highly recommended) ---
    # ---------------------------------------------------------------------

    # --------------------------------------------------------------------------------- __str__()
    def __str__(self) -> str:
        """Returns a user-friendly string representation of the [MyClassName] object.

        This method is primarily for end-user display (e.g., `print(object)`).
        """
        return "[A human-readable description of the object's state or summary.]"
    # --------------------------------------------------------------------------------- end __str__()

    # --------------------------------------------------------------------------------- __repr__()
    def __repr__(self) -> str:
        """Returns an unambiguous string representation of the [MyClassName] object.

        This method is primarily for developers and debugging. The goal is that
        `eval(repr(obj))` should ideally recreate the object.
        """
        return f"[MyClassName]('{self.public_property}')" # Example, adapt to your constructor
    # --------------------------------------------------------------------------------- end __repr__()

# ------------------------------------------------------------------------- end class MyClassName   


# =========================================================================
# Standalone Function Definitions
# =========================================================================
# These are functions that are not methods of any specific class within this module.

# --------------------------
# --- Utility Functions ---
# --------------------------

# --------------------------------------------------------------------------------- my_utility_function()
def [my_utility_function]([param1]: [type]) -> [return_type]:
    """[A concise one-line summary of the function's purpose.]

    [Provide a more detailed explanation of what the function does.]
    [Describe its algorithm, any side effects, or important considerations.]

    Args:
        [param1] ([type]): [Description of param1.]
        [param2] ([type], optional): [Description of param2.] Defaults to [default_value].

    Returns:
        [return_type]: [Description of the value returned by the function.]

    Raises:
        [ExceptionType]: [Condition under which this exception is raised.]

    Examples:
        >>> [my_utility_function]([example_input])
        [expected_output]
    """
    # [Function implementation logic goes here]
    # Use inline comments for complex or non-obvious lines of code.
    # e.g., # This calculation ensures data normalization.
    pass
# --------------------------------------------------------------------------------- end my_utility_function()

# ------------------------
# --- Helper Functions ---
# ------------------------

# --------------------------------------------------------------------------------- get_user_confirmation()
def get_user_confirmation(prompt: str) -> bool:
    """Prompts the user for a yes/no confirmation.

    Args:
        prompt (str): The question to ask the user.

    Returns:
        bool: True if the user confirms (Y/y), False otherwise (N/n).
    """
    while True:
        choice = input(f"{prompt} [Y/N]: ").lower()
        if choice in ('y', 'yes'):
            return True
        elif choice in ('n', 'no'):
            return False
        else:
            print("Invalid input. Please enter 'Y' or 'N'.")
# --------------------------------------------------------------------------------- end get_user_confirmation()

# ---------------------------------------------
# --- Callable Functions from other modules ---
# ---------------------------------------------

# --------------------------------------------------------------------------------- process_module_data()
def process_module_data(data: list) -> dict:
    """Processes a list of data relevant to this module's scope.

    This function demonstrates a typical standalone function that might be
    called from another module's main logic.

    Args:
        data (list): The list of data items to process.

    Returns:
        dict: A dictionary of aggregated results.
    """
    # [Module-specific data processing logic]

    # -----------------------
    # -- Embedded Classes --
    # ----------------------

    # --------------------------------------------------------------------------------- my_embedded_function()
    def [my_embedded_function]()
        # embedded my_embedded_function body logic
        return ...
    # --------------------------------------------------------------------------------- end my_embedded_function()
    
    return {"processed_count": len(data)}
# --------------------------------------------------------------------------------- end process_module_data()

# =========================================================================
# Module Initialization / Main Execution Guard (if applicable)
# =========================================================================
# This block runs only when the file is executed directly, not when imported.
# For a file part of a larger program, it typically contains
# module-specific tests or example usage. It should *not* contain the main
# application logic, which belongs in the project's primary entry point (e.g., main.py).

if __name__ == '__main__':
    # --- Example Usage / Module-specific Tests ---
    print(f"Running tests for {__file__}...")

    # Example: Test a class defined in this module
    # try:
    #     my_test_object = [MyClassName]("test_value")
    #     print(f"Test object created: {my_test_object!r}") # Use !r for __repr__
    #     # Add specific test calls here
    #     # result = my_test_object.add_item({"id": 1, "name": "Test Item"})
    #     # print(f"Add item test: {result}")
    # except Exception as e:
    #     print(f"Error during module test: {e}")

    # Example: Test a standalone function
    # test_result = [my_utility_function](10)
    # print(f"Utility function test result: {test_result}")

    print(f"Finished tests for {__file__}.")   
# ---------------------------------------------------------------------------------

# =========================================================================
# End of File
# =========================================================================

```