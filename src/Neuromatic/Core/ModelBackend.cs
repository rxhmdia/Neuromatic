using Neuromatic.Initializers;
using Neuromatic.Layers;
using System;
using System.Collections.Generic;
using System.Text;

namespace Neuromatic.Core
{
    /// <summary>
    /// Visitor used to compile the model to an executable graph
    /// </summary>
    public abstract class ModelBackend : IDisposable
    {
        /// <summary>
        /// Visits an input layer
        /// </summary>
        /// <param name="layer">Input layer to visit</param>
        public abstract ExecutableModelNode Placeholder(string name, long[] shape);

        /// <summary>
        /// Locates a node in the graph that is managed by the backend
        /// </summary>
        /// <param name="name">Name of the model node</param>
        /// <returns>Returns the found model node. Returns null when the node could not be found.</returns>
        public abstract ExecutableModelNode Node(string name);

        /// <summary>
        /// Create a node that performs matrix multiplication
        /// </summary>
        /// <param name="left">Left node</param>
        /// <param name="right">Right node</param>
        /// <param name="name">Name of the node</param>
        /// <returns>Returns the matrix multiplication node</returns>
        public abstract ExecutableModelNode Dot(ExecutableModelNode left, ExecutableModelNode right, string name);

        /// <summary>
        /// Initializes a tensor with a normal distribution
        /// </summary>
        /// <param name="shape">Shape of the tensor</param>
        /// <param name="mean">Mean value for the distribution</param>
        /// <param name="standardDeviation">Standard deviation</param>
        /// <param name="seed">Random seed to use</param>
        /// <returns></returns>
        public abstract ExecutableModelNode RandomNormal(long[] shape, float mean, float standardDeviation, int? seed);

        /// <summary>
        /// Creates a new set of weights for the model
        /// </summary>
        /// <param name="initializer">Initializer to use</param>
        /// <param name="name">Name of the node</param>
        /// <returns>Returns the model node for the weights</returns>
        public abstract ExecutableModelNode Weights(long[] shape, InitializationFunction initializer, string name);

        /// <summary>
        /// Adds a bias term to the given model node
        /// </summary>
        /// <param name="node">Node to add the bias term to</param>
        /// <param name="bias">Bias term to add</param>
        /// <returns>Returns the new model node with the bias term added</returns>
        public abstract ExecutableModelNode BiasAdd(ExecutableModelNode node, ExecutableModelNode bias);

        /// <summary>
        /// Creates a new constant value
        /// </summary>
        /// <param name="value">Value for the constant</param>
        /// <param name="name">Name of the constant</param>
        /// <returns>Returns the node for the constant value</returns>
        public abstract ExecutableModelNode Constant(object value, long[] shape, string name);

        /// <summary>
        /// Creates a functions that adds two inputs
        /// </summary>
        /// <param name="a">First input</param>
        /// <param name="b">Second input</param>
        /// <param name="name">Name of the operation</param>
        /// <returns>Returns the node for the add operation</returns>
        public abstract ExecutableModelNode Add(ExecutableModelNode a, ExecutableModelNode b, string name = null);

        /// <summary>
        /// Multiplies two values
        /// </summary>
        /// <param name="left">Left variable</param>
        /// <param name="right">Right variable</param>
        /// <returns>Returns the output of the multiplication operation</returns>
        public abstract ExecutableModelNode Multiply(ExecutableModelNode left, ExecutableModelNode right);

        /// <summary>
        /// Creates a function bound to the backend
        /// </summary>
        /// <param name="defaultInputs">The default values for all placeholders in the function</param>
        /// <param name="outputs">The list of outputs to fetch as part of the function</param>
        /// <returns>Returns an executable function</returns>
        public abstract BackendFunction Function(IEnumerable<ExecutableModelNode> inputs, IEnumerable<ExecutableModelNode> outputs, IEnumerable<ExecutableModelNode> updates);

        /// <summary>
        /// Defines a functions that determines the gradients of the loss in relation to the variables.
        /// </summary>
        /// <param name="loss">Loss function</param>
        /// <param name="variables">Variables for the loss function</param>
        /// <returns>Returns the gradients for the loss in relation to the variables</returns>
        public abstract IEnumerable<ExecutableModelNode> Gradients(ExecutableModelNode loss, IEnumerable<ExecutableModelNode> variables);

        /// <summary>
        /// Defines a variable with an integer value
        /// </summary>
        /// <param name="value">Initial value for the variable</param>
        /// <param name="name">Name of the variable</param>
        public abstract ExecutableModelNode Variable(int value, string name);

        /// <summary>
        /// Updates a variable by adding a value to it
        /// </summary>
        /// <param name="variable">Variable to update</param>
        /// <param name="value">Value to add</param>
        public abstract ExecutableModelNode UpdateAdd(ExecutableModelNode variable, ExecutableModelNode value);

        /// <summary>
        /// Updates a variable with a new value
        /// </summary>
        /// <param name="w">Variable to update</param>
        /// <param name="newValue">The new value to assign to the variable</param>
        /// <returns></returns>
        public abstract ExecutableModelNode Update(ExecutableModelNode original, ExecutableModelNode newValue);

        /// <summary>
        /// Creates a sigmoid function
        /// </summary>
        /// <param name="input">Input for the function</param>
        /// <returns>Output of the function</returns>
        public abstract ExecutableModelNode Sigmoid(ExecutableModelNode input);

        /// <summary>
        /// Clips the input by a minimum and maximum value
        /// </summary>
        /// <param name="input">Input to clip</param>
        /// <param name="minValue">Min value to clip to</param>
        /// <param name="maxValue">Max value to clip to</param>
        /// <returns>Returns the clipped value</returns>
        public abstract ExecutableModelNode ClipByValue(ExecutableModelNode input, float minValue, float maxValue);

        /// <summary>
        /// Gets the trainable weights
        /// </summary>
        public abstract IEnumerable<ExecutableModelNode> TrainableWeights { get; }

        /// <summary>
        /// Subtract two values
        /// </summary>
        /// <param name="left">Left node</param>
        /// <param name="right">Right node</param>
        /// <returns>Returns the subtraction operator</returns>
        public abstract ExecutableModelNode Subtract(ExecutableModelNode left, ExecutableModelNode right);

        /// <summary>
        /// Performs a natural log on the input
        /// </summary>
        /// <param name="input"></param>
        /// <param name="subtract"></param>
        /// <returns></returns>
        public abstract ExecutableModelNode Log(ExecutableModelNode input);

        /// <summary>
        /// Performs a division operation
        /// </summary>
        /// <param name="numerator"></param>
        /// <param name="divisor"></param>
        /// <returns></returns>
        public abstract ExecutableModelNode Divide(ExecutableModelNode numerator, ExecutableModelNode divisor);

        /// <summary>
        /// Calculates the sigmoid cross-entropy with logits
        /// </summary>
        /// <param name="target">Target tensor</param>
        /// <param name="result">Result tensor</param>
        /// <returns>Returns the outcome of the operation</returns>
        public abstract ExecutableModelNode SigmoidCrossEntropyWithLogits(
            ExecutableModelNode target, ExecutableModelNode result);

        /// <summary>
        /// Generates a cast operation to float
        /// </summary>
        /// <param name="input">The input to cast</param>
        /// <returns>Returns the cast operator</returns>
        public abstract ExecutableModelNode Float(ExecutableModelNode input);
        
        #region IDisposable Support

        /// <summary>
        /// Disposes any resources used by the backend
        /// </summary>
        /// <param name="disposing"></param>
        protected virtual void Dispose(bool disposing)
        {

        }

        // This code added to correctly implement the disposable pattern.
        public void Dispose()
        {
            Dispose(true);
        }

        #endregion
    }
}
