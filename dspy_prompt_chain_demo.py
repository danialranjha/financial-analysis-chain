import os
import dspy

# Configure the LM globally
lm = dspy.LM("openai/gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
dspy.configure(lm=lm)

def prompt_chain(input_text):
    # First step: ask for a company overview
    step1 = dspy.Predict("question: str -> answer: str")
    result1 = step1(question=input_text)
    answer1 = getattr(result1, "answer", result1)

    # Second step: summarize the overview
    step2 = dspy.Predict("question: str -> answer: str")
    result2 = step2(question=f"Summarize: {answer1}")
    answer2 = getattr(result2, "answer", result2)

    return answer2

if __name__ == "__main__":
    final = prompt_chain("Tell me about NVIDIA")
    print("Final output:", final)