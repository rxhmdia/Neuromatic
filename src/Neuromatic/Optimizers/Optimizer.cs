using Neuromatic.Core;
using System;
using System.Collections.Generic;
using System.Text;

namespace Neuromatic.Optimizers
{
    /// <summary>
    /// Inherit from this class to implement a model optimizer routine
    /// </summary>
    public abstract class Optimizer
    {
        /// <summary>
        /// Compiles the optimizer to a backend specific implementation
        /// </summary>
        /// <param name="backend"></param>
        public abstract void Compile(ModelBackend backend);
    }
}
