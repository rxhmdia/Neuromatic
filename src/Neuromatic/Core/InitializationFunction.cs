using System;
using System.Collections.Generic;
using System.Text;

namespace Neuromatic.Core
{
    /// <summary>
    /// Wrapper around an initialization function
    /// </summary>
    public class InitializationFunction
    {
        private Func<long[], ExecutableModelNode> _functor;

        /// <summary>
        /// Initializes a new instance of <see cref="InitializationFunction"/>
        /// </summary>
        /// <param name="functor"></param>
        public InitializationFunction(Func<long[], ExecutableModelNode> functor)
        {
            _functor = functor;
        }

        /// <summary>
        /// Creates the actual initialization function
        /// </summary>
        /// <returns>Returns the reference to the model node that implements the initialization function</returns>
        public ExecutableModelNode Create(long[] shape)
        {
            return _functor(shape);
        }
    }
}
