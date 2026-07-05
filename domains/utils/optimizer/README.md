# Optimizer
> **English** | [繁體中文](./README.zh-TW.md)

In machine learning (especially deep learning), the "optimizer" plays a crucial role.

### What is an optimizer?

In simple terms, an optimizer is an **algorithm** or **mechanism** whose **sole purpose** is to **update** the parameters of a neural network (i.e. the weights $W$ and biases $b$) in order to **minimize the "loss function"**.


### Core Concepts Explained
1.  **Goal: minimize the loss (Loss)**
    * **Loss Function:** like the model's "exam grader". It measures the **gap** between the model's current prediction and the "correct answer". The larger the gap, the higher the loss value (Loss).
    * **The optimizer's goal:** to find a set of **best parameters** ($W$ and $b$) that drive this "loss value" as **low** as possible.
2.  **Method: how to update the parameters?**
    * **Gradient:** this is the optimizer's most important tool. The gradient tells the model: "If you want to reduce the loss, in which direction should you adjust your parameters?" The gradient points in the direction in which the loss **increases fastest**.
    * **Update rule:** the optimizer takes this gradient and then **operates in reverse** (i.e. "gradient descent"). It makes a small adjustment to the model's parameters along the direction **opposite** to the gradient.
    * **Learning Rate:** this is like the "step size" of the adjustment. Too large a step may overshoot the minimum; too small a step makes learning too slow.

3.  **A simple analogy: descending a mountain blindfolded**
    * **You (the model):** standing on a large mountain (the terrain of the loss function), with your eyes blindfolded.
    * **Your position (parameters):** where you currently stand on the mountain.
    * **The valley (the minimum):** your goal, i.e. the place where the loss (Loss) is smallest.
    * **Your feet (the gradient):** you use your feet to feel the "slope" around you (i.e. the gradient), figuring out which direction is the "steepest downhill path".
    * **The optimizer:** it is the "strategy" by which you **decide how to descend**.
    * **The learning rate:** how big **each step** you decide to take.

### Why are there so many kinds of optimizers?
* **SGD (Standard Gradient Descent):** simply follows the current slope. If it hits a flat region (gradient of 0), it may just stop there.
* **Momentum:** like a ball rolling down the mountain. It considers not only the current slope but also **accumulates past velocity (momentum)**. This helps it charge through flat regions and reach the valley faster.
* **Adam:** an **adaptive** strategy. It keeps a separate learning rate (step size) for **each parameter**, and also accumulates momentum like Momentum does.

**Summary:**
The optimizer is the core algorithm responsible for "**how to learn**". By continually looking at the "exam score" (loss) and using the "gradient" (downhill direction), it gradually adjusts the model's parameters to eventually find the set of parameters that scores highest (lowest loss).