# Repository Engineering Rules

## GAN discriminator safety

- Treat discriminator normalization as part of the training contract. Before
  changing a GAN loss or discriminator call site, inspect whether the
  discriminator contains `BatchNorm` or another batch-dependent layer.
- Never concatenate real and generated samples into one discriminator forward
  and split the logits afterward when their positions encode the class label.
  Run separate forwards for real and generated samples instead:

  ```python
  logits_real = discriminator(real.detach())
  logits_fake = discriminator(fake.detach())
  ```

- Preserve this rule in conditional GAN paths as well: build the real and fake
  conditioned inputs separately, then call the discriminator separately.
- Any GAN implementation or refactor must retain or add a regression test that
  rejects the combined-batch pattern. Run `tests/test_gan_discriminator_contract.py`
  and the relevant training-contract tests before committing.
- Do not work around this contract by disabling an existing test. A deliberate
  normalization redesign requires an explicit explanation and replacement test.
