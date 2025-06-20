import java.util.*;

public class Main {

    public static void main(String[] args){

        String test = "abcabcbb";
        int GetMax = lengthOfLongestSubstring(test);
        System.out.println(GetMax);

    }

    public static int lengthOfLongestSubstring(String s) 
    {
        HashMap<Character, Integer> tracker = new HashMap<>();
        int maxlength = 0;
        int left = 0;

        for (int right = 0; right < s.length(); right++)
        {
            char current = s.charAt(right);

            if(tracker.containsKey(current))
            {
                left = Math.max(tracker.get(current)+1, left);
            }
            tracker.put(current, right);
            maxlength = Math.max(maxlength, right - left + 1);


        }
        return maxlength;


        
    }


    // Solved Partition problem using a Greedy Approach 
    public static int partitionArray(int[] nums, int k) {
        Arrays.sort(nums);

        int sub_count=0;
        int min_val = -1;

        for(int num: nums){

            if(sub_count == 0 || (num - min_val) > k){
                sub_count++;
                min_val = num;
            }


        }
        return sub_count;
        
    }

    // Solved Content Children (Cookies) using the greedy approach (greedy choice: least greedy child)
    public static int ContentChildren(int[] g, int[] s){
        Arrays.sort(g);
        Arrays.sort(s);
        int st_idx = 0;
        int c_idx = 0;
        int give_c = 0;

        while (st_idx < g.length && c_idx < s.length){
            if (s[c_idx] >= g[st_idx]){
                give_c++;
                st_idx++;
            }
            c_idx++;
        }
        return give_c;
    }

 

    





    
}
