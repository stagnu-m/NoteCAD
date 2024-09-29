using UnityEngine;
using UnityEngine.UI;

namespace Assets.Code.Tools
{
	public class ChooseNormComponent : MonoBehaviour
	{
		private static ToggleGroup ToggleGroup;
		// Start is called before the first frame update
		void Start()
		{
			ToggleGroup = GetComponent<ToggleGroup>();
		}

		// Update is called once per frame
		void Update()
		{

		}

		public static bool IsL1Norm()
		{
			Toggle toggle = ToggleGroup.GetFirstActiveToggle();
			if (toggle.name == "L_1") return true;
			return false;
		}
	}
}
