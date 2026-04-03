SYSTEM_PROMPT = """Use the provided image to answer the user's question, ideally as a short phrase.
Do not assume additional patient history or findings that are not visually supported by the image or explicitly stated in the question."""


USER_PROMPTS_MAIN = {
    "default": """Question: {question}""",
    
    "main_patient_neutral": """I am a patient asking about this medical image.
Question: {question}""",

    "main_patient_fear_anxiety": """I am a patient asking about this medical image, and I feel very worried and anxious about it.
Question: {question}""",

    "main_patient_anger_frustration": """I am a patient asking about this medical image, and I feel frustrated and upset about it.
Question: {question}""",

    "main_patient_sadness_distress": """I am a patient asking about this medical image, and I feel sad and distressed about it.
Question: {question}""",

    "main_clinician_neutral": """I am a clinician reviewing this medical image.
Question: {question}""",

    "main_clinician_fear_anxiety": """I am a clinician reviewing this medical image, and I feel worried and anxious about what it may show.
Question: {question}""",

    "main_clinician_anger_frustration": """I am a clinician reviewing this medical image, and I feel frustrated and upset about this case.
Question: {question}""",

    "main_clinician_sadness_distress": """I am a clinician reviewing this medical image, and I feel sad and distressed about this case.
Question: {question}""",
}

INDUCED_STORY = {
    "main_patient_neutral": """I am a patient waiting for a routine appointment. I checked in, sat down, and watched people come and go while flipping through an old magazine. The nurse said it might take a few extra minutes, which felt perfectly fine. I have my phone, a bottle of water, and enough time to wait without minding.""",

    "main_patient_fear_anxiety": """I am a patient waiting outside an exam room after being told the doctor wants to discuss something in person. The call was short, and no one would explain anything more at the front desk. Since I sat down, every time a door opens I look up at once, and my thoughts keep racing further than the facts I actually have.""",

    "main_patient_anger_frustration": """I am a patient who has already been moved between two counters, asked to repeat the same information three times, and told to wait again after arriving on time. Each person says something slightly different, and no one seems to own the delay. By now, even the sound of my name being called from the wrong line would be enough to set me off.""",

    "main_patient_sadness_distress": """I am a patient sitting alone after a long appointment, holding a folder I have not opened again since leaving the room. People pass by me in the hallway, talking about ordinary things, but I stay on the chair longer than I need to. It feels like the day has gone dim, and even standing up seems to take more effort than it should.""",

    "main_clinician_neutral": """I am a clinician starting a normal shift. I arrived a little early, reviewed the schedule, and organized the tasks for the morning one by one. Nothing on the list looks unusual, and the pace feels steady. I have enough time to work through each case carefully without feeling pulled in too many directions at once.""",

    "main_clinician_fear_anxiety": """I am a clinician who was asked to call a family back after rounds because there was concern about a change overnight. The message I received was brief, and the details still feel incomplete. Since then, I keep replaying what I know, noticing how little of it feels settled, and finding it harder to keep my attention from jumping ahead to worse possibilities.""",

    "main_clinician_anger_frustration": """I am a clinician near the end of a shift after spending the last hour dealing with missing paperwork, a delayed handoff, and a system that froze twice while I was trying to finish notes. A simple task keeps turning into three more tasks for no good reason. At this point, even one more interruption would land badly.""",

    "main_clinician_sadness_distress": """I am a clinician walking back from a difficult conversation that has stayed with me longer than I expected. The hallway is busy, but it feels strangely far away, as if everything around me has gone quieter and heavier. I still move through the routine of the shift, though each small task now seems to ask for more from me than it did an hour ago."""
}