using System;
using System.Collections.Generic;
using System.Text;
using Neuromatic.Activations;

namespace Neuromatic
{
    /// <summary>
    /// Defines several out-of-the-box activation functions
    /// </summary>
    public static class StandardActivations
    {
        /// <summary>
        /// Creates a new sigmoid activation function
        /// </summary>
        /// <returns></returns>
        public static ActivationFunction Sigmoid()
        {
            return new Sigmoid();
        }
    }
}
