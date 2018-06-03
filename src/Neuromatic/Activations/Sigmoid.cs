using System;
using System.Collections.Generic;
using System.Text;
using Neuromatic.Core;

namespace Neuromatic.Activations
{
    /// <summary>
    /// The sigmoid activation function.
    /// </summary>
    public class Sigmoid : ActivationFunction
    {
        /// <summary>
        /// Compiles the activation function
        /// </summary>
        /// <param name="input">Input for the function</param>
        /// <param name="backend">Backend to use for compilation</param>
        /// <returns>Output of the activation function</returns>
        public override ExecutableModelNode Compile(ExecutableModelNode input, ModelBackend backend)
        {
            return backend.Sigmoid(input);
        }
    }
}
