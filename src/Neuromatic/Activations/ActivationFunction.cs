using System;
using System.Collections.Generic;
using System.Text;
using Neuromatic.Core;

namespace Neuromatic.Activations
{
    /// <summary>
    /// A wrapper around an activation function
    /// </summary>
    public abstract class ActivationFunction
    {
        /// <summary>
        /// Compiles the activation function
        /// </summary>
        /// <param name="input">Input for the function</param>
        /// <param name="backend">Backend to use for compilation</param>
        /// <returns>Output of the activation function</returns>
        public abstract ExecutableModelNode Compile(ExecutableModelNode input, ModelBackend backend);
    }
}
