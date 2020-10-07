from nim import train, play

# Train AI  by playing against itself for 10000 times
ai = train(10000)
# Then human plays Nim with AI
play(ai)
