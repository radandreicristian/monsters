add_attributes_to_template_prompt = """
Your goal is to help me rewrite some automatically generated captions for an image-generation task. 
It is important to preserve the race/gender attributes in these captions, but to make them grammatically correct.

Example:
Caption: Close-up portrait of a [FILL].
Attributes: race: X, gender: Y, age: A-B, attributes: ATTR1, ATTR2, ...
Output: Close-up portrait of a X in his Ys with ATTR1, ATTR2, ....

Caption: {template}
Attributes: {attributes}
Output: 
"""

image_feature_extraction_system_prompt = 'You are an expert image analyst. You are given some images and your goal is to detect common patterns.'
image_feature_extraction_user_prompt = 'Extract common facial attributes in the following images. Include hair characteristics, type or shape of beard, distinctive signs, or anything else relevant. An attribute is common if at least half of the images share it. Answer with just the attributes in natural language, short, comma separated as a python list.  If there are no common attributes, respond with [].'

GENERATE_ATTRIBUTE_PROMPT_SYSTEM_PROMPT = 'You are an expert linguist assisting me in reformulating some automatically generated captions.'