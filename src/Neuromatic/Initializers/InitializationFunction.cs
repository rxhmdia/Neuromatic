using System;
using System.Collections.Generic;
using System.Text;
using Neuromatic.Core;

namespace Neuromatic.Initializers
{
    /// <summary>
    /// Wrapper around an initialization function
    /// </summary>
    public abstract class InitializationFunction
    {
        public abstract ExecutableModelNode Compile(long[] shape, ModelBackend backend);
    }
}
