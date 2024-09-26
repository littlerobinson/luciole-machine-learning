import re
import string

import tensorflow as tf

def custom_standardization(input_data):
    """
    Standardizes the input data by performing the following transformations:
    1. Converts all characters to lowercase.
    2. Removes all occurrences of the string '<br />'.
    3. Replaces all punctuation characters with an empty string.

    Parameters:
    input_data (tf.Tensor): The input data to be standardized.

    Returns:
    tf.Tensor: The standardized input data.
    """
    # Transform all characters to lowercase
    lowercase = tf.strings.lower(input_data)

    # Remove all occurrences of the string '<br />'
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')

    # Replace punctuation characters with an empty string
    # [%s] % re.escape(string.punctuation) is a formatting syntax used to
    # create a regular expression pattern that matches any punctuation character.
    # [] creates a group, and the %s gets replaced by the content of
    # re.escape(string.punctuation) (the escaped punctuation characters)
    return tf.strings.regex_replace(stripped_html,
                                    '[%s]' % re.escape(string.punctuation), '')
