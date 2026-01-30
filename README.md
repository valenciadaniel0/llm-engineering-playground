“LLMs are infrastructure. The engineering challenge is integrating them into reliable, scalable systems.”

## Prompt Control Experiments
*What temperature changed*
It didn't changed much because the prompt was very specific and simple, probably asking it to create something is going to make it more creative with a higher temperature.

*What system prompts improved*
didn't see important differences or improvements on the response, probably tailoring the prompt to the role is going to give us better results, but with this simple prompt it doesn't change much.

*One surprise you observed*
No surprises , it was pretty deterministic.

## Cost Awareness
Cost per request: 0.0001725
Cost per 1k requests: 0.1725
cost-control idea: Limit the input lenght for the users, and make sure to pass system configurations to limit the response to 1 paragraph.
