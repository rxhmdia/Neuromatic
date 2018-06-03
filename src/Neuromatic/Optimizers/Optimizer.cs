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
        /// Compiles the optimizer into a set of operations to be executed in each minibatch
        /// </summary>
        /// <param name="loss">Loss function to optimize</param>
        /// <param name="weights">The trainable model weights</param>
        /// <param name="backend">The backend used for compilation</param>
        /// <returns>Returns the compiled set of operations for the optimizer</returns>
        public abstract IEnumerable<ExecutableModelNode> Compile(ExecutableModelNode loss, IEnumerable<ExecutableModelNode> weights, ModelBackend backend);
    }
}
