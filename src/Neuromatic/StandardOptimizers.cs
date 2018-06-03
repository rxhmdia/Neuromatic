using Neuromatic.Optimizers;
using System;
using System.Collections.Generic;
using System.Text;

namespace Neuromatic
{
    /// <summary>
    /// Defines a set of standard optimizers
    /// </summary>
    public static class StandardOptimizers
    {
        /// <summary>
        /// Creates a Stochastic Gradient Descent optimizer
        /// </summary>
        /// <param name="learningRate">The learning rate for the optimzer</param>
        /// <returns>Returns the optimizer instance</returns>
        public static Optimizer SGD(float learningRate = 0.01f)
        {
            return new SGD(learningRate);
        }
    }
}
