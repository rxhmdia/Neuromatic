using Neuromatic.Losses;
using System;
using System.Collections.Generic;
using System.Text;

namespace Neuromatic
{
    /// <summary>
    /// Defines a set of standard loss functions available in the framework
    /// </summary>
    public static class StandardLosses
    {
        /// <summary>
        /// Creates the binary cross entropy loss function
        /// </summary>
        /// <returns></returns>
        public static LossFunction BinaryCrossEntropy()
        {
            return new BinaryCrossEntropy();
        }
    }
}
