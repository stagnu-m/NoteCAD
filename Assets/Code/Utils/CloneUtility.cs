using System;
using System.Collections.Generic;

namespace Assets.Code.Utils
{
	public static class CloneUtility
	{
		public static T DeepClone<T>(T obj)
		{
			var clonedObjects = new Dictionary<object, object>();
			return (T)CloneInternal(obj, clonedObjects);
		}

		private static object CloneInternal(object obj, Dictionary<object, object> clonedObjects)
		{
			if (obj == null) return null;

			switch (obj)
			{
				case Param paramObj:
					return paramObj.Clone(clonedObjects);
				case Exp expObj:
					return expObj.Clone(clonedObjects);
				default:
					throw new InvalidOperationException($"Unsupported type: {obj.GetType()}");
			}
		}
	}

}