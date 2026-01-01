from qwen import QwenLocal


def main():
    llm = QwenLocal()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a TEI Guidelines tutor. "
                "You explain rules clearly and do not invent standards."
            ),
        },
        {
            "role": "user",
            "content": "Explain when <persName> should be used in TEI.",
        },
    ]

    response = llm.chat(
        messages,
        temperature=0.2,
        maxNewTokens=300,
    )

    print("\n--- MODEL OUTPUT ---\n")
    print(response)


if __name__ == "__main__":
    main()