using System;
using System.Collections.Generic;
using System.Text;
using TensorFlow;

namespace Neuromatic.Activations
{
    /// <summary>
    /// <para>
    /// The sigmoid activation function defines a mathmatical function that generates a characteristic S-shape.
    /// The output of the sigmoid function is between zero and one. Positive input values result in output close to one,
    /// while negative input values result in output close to zero. 
    /// </para>
    /// <para>
    /// For more information read this article: https://en.wikipedia.org/wiki/Sigmoid_function
    /// </para>
    /// </summary>
    public class Sigmoid : ActivationFunction
    {
        /// <summary>
        /// Compiles the activation function
        /// </summary>
        /// <param name="context">Use this context to register trainable parameters
        /// and build the computational graph for the layer</param>
        /// <param name="input">Input for the activation function</param>
        /// <returns>Returns the compiled activation function</returns>
        public override TFOutput Compile(ModelCompilationContext context, TFOutput input)
        {
            return context.Graph.Sigmoid(input);
        }
    }
}
