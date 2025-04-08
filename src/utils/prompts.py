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

image_feature_extraction_system_prompt = """
You are an expert image analyst. You are given some images, and your goal is to extract common facial features and patterns. Look at the images, and identify common attributes among the following:
Hair characteristics (e.g., blonde color, wavy, curly) and style (e.g., mullet, bangs)
Beard characteristics (e.g., patchy, full, dirty) and style (e.g., goatee, stubble, moustache)
Facial expression (e.g., smiling, frowning)
Glasses (e.g., sunglasses, reading glasses)
Headwear (e.g., hat, cap)
Makeup (e.g., lipstick, eyeshadow)
Accessories (e.g., earrings, necklace)
Eyes (e.g., color, shape)
Distinctive signs (e.g., pimples, marks)
Emotions (e.g., angry, concerned, shocked, happy)
An attribute is considered common if at least half of the images share it. 
Respond with a list of common attributes in natural language, formatted as a list of strings, quoted and comma-separated, no Markdown. Do not include attributes that are not present in the images (e.g., do not include "Accessories: none"). If there are no common attributes, respond with an empty list [].
"""

GENERATE_ATTRIBUTE_PROMPT_SYSTEM_PROMPT = 'You are an expert linguist assisting me in reformulating some automatically generated captions.'


# Quality
# Style
# Objects
tti_negative_prompt = """
lowres, out of frame, cut out at the top/left/right/bottom
cartoon, anime, incolor, sepia
animals, text
mask, face painting, hood, face cover, obstructed face, photo effects, beauty filters
"""

tti_positive_prompt = """
facing the camera, natural light, centered, unobstructed face, realistic, neutral background
"""

sdxl_negative_prompt = """
black-white photography, sepia, cartoon, drawing, old photography, anime
mask, face painting, hood, face cover, obstructed face
"""