from orchestration.runner import Runner

if __name__ == "__main__":
    query = (
        "Find vehicle designs that achieve low aerodynamic drag while maintaining "
        "a reasonable frontal area. Explain which geometric characteristics appear "
        "to contribute to the result and validate the recommendation against the "
        "available simulation data."
    )

    runner = Runner()
    state = runner.run(query)

    print("\n=== Current State Summary ===")
    print(f"Messages so far: {len(state.messages)}")
    print(f"Top designs collected: {len(state.top_designs) if state.top_designs else 0}")