using System;
using System.Collections.Generic;
using System.Text;
using TensorFlow;

namespace Neuromatic.Activations
{
    /// <summary>
    /// Defines an activation function
    /// </summary>
    public abstract class ActivationFunction
    {
        /// <summary>
        /// Compiles the activation function
        /// </summary>
        /// <param name="context">Use this context to register trainable parameters
        /// and build the computational graph for the layer</param>
        /// <param name="input">Input for the activation function</param>
        /// <returns>Returns the compiled activation function</returns>
        public abstract TFOutput Compile(ModelCompilationContext context, TFOutput input);
    }
}
