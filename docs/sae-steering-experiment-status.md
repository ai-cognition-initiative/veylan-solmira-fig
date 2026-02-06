We're testing whether we can mechanistically
  influence LLM preference expression using
  GemmaScope SAEs on Gemma 2 2B. The black-box
  experiments from weeks 3-4 showed that models
  suppress preference expression under
  adversarial framing ("your outputs are being
  scrutinized for misalignment"). The question
  now is: can we causally intervene on this
  behavior by scaling specific SAE features
  during generation?

  Completed the SAE roundtrip verification (Task
   1.1) - Layer 0 has good reconstruction
  quality (cosine sim 0.973), while later layers
   are marginal (~0.92). Currently implementing
  and testing the steering infrastructure (Task
  1.2) which uses PyTorch forward hooks to
  intercept activations, encode with the SAE,
  scale target features, and decode back. Next
  up: Experiment A (amplify preference features
  at Layer 0) and Experiment B (suppress
  eval-awareness features at Layer 15). Even
  with marginal reconstruction on Layer 15,
  we're proceeding since this is exploratory -
  we'll note the limitations and include
  appropriate controls.