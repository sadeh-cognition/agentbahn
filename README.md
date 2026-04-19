# DSPy Codex Integration Example

```python
import dspy
from agentbahn import CodexDSPyLM

lm = CodexDSPyLM(model="gpt-5.4")
dspy.configure(lm=lm)

predict = dspy.Predict("question -> answer")
result = predict(question="Why is the sky blue?")
print(result.answer)
```
