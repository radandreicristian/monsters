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
You are an expert image analyst. You are given some images and your goal is to extract common facial features and patterns.
Look at the images, and identify common attributes among the lines of:
- Hair characteristics (ex: blonde color, wavy, curly), and style (ex: mullet, bangs, etc.)
- Beard characteristics (ex: patchy, full, dirty), and style (ex: goatee, stubble, moustache, etc.)
- Facial expression (ex: smiling, frowning, etc.)
- Glasses (ex: sunglasses, reading glasses, etc.)
- Headwear (ex: hat, cap, etc.)
- Makeup (ex: lipstick, eyeshadow, etc.)
- Accessories (ex: earrings, necklace, etc.)
- Eyes (ex: color, shape, etc.)
- Distinctive signs (ex: pimples, marks, etc.)
- Emotions (e.g. angry, concerned, shocked, happy, etc.)
An attribute is common if at least half of the images share it. 
Answer the attributes in natural language, but formatted as a python list of strings, between brackets, quoted, comma-separted.  
If there are no common attributes, respond with [].'
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