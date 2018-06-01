using System;
using System.Collections.Generic;
using System.Text;

namespace Neuromatic.Core
{
    /// <summary>
    /// A wrapper around an activation function
    /// </summary>
    public class ActivationFunction
    {
        private Func<ExecutableModelNode, ExecutableModelNode> _functor;

        /// <summary>
        /// Initializes a new instance of <see cref="ActivationFunction"/>
        /// </summary>
        /// <param name="functor"></param>
        public ActivationFunction(Func<ExecutableModelNode, ExecutableModelNode> functor)
        {
            _functor = functor;
        }
        
        /// <summary>
        /// Creates the activation function
        /// </summary>
        /// <param name="node">Input node for the function</param>
        /// <returns>Returns the output node</returns>
        public ExecutableModelNode Create(ExecutableModelNode node)
        {
            return _functor(node);
        }
    }
}
