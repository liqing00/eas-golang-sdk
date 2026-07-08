package eas

import (
	"fmt"
	"net/url"
	"testing"
	"time"
)

const (
	EndpointName    = ""
	PMMLName        = "scorecard_pmml_example"
	PMMLToken       = ""
	TensorflowName  = "mnist_saved_model_example"
	TensorflowToken = ""
	TorchName       = "pytorch_resnet_example"
	TorchToken      = ""
	TestName        = "test_example"
	TestToken       = ""
)

func TestString(t *testing.T) {

	client := NewPredictClient(EndpointName, PMMLName)
	client.SetToken(PMMLToken)
	client.Init()
	req := "[{}]"
	client.AddHeader("headerName", "headerValue")
	resp, err := client.StringPredict(req)
	if err != nil {
		t.Fatalf(err.Error())
	} else {
		fmt.Printf("%v\n", resp)
	}
}

func TestTF(t *testing.T) {
	cli := NewPredictClient(EndpointName, TensorflowName)
	cli.SetToken(TensorflowToken)
	cli.Init()

	req := TFRequest{}
	req.SetSignatureName("predict_images")
	req.AddFeedFloat32("images", []int64{1, 784}, make([]float32, 784))

	st := time.Now()
	for i := 0; i < 10; i++ {
		resp, err := cli.TFPredict(req)
		if err != nil {
			t.Fatalf("failed to query tf model: %v", err)
		}
		fmt.Printf("%v\n", resp)
	}

	fmt.Println("average response time : ", time.Since(st)/10)
}

// TestTorch tests pytorch request and response unit test
func TestTorch(t *testing.T) {

	cli := NewPredictClient(EndpointName, TorchName)
	cli.SetTimeout(500)
	cli.SetToken(TorchToken)
	cli.Init()
	req := TorchRequest{}
	req.AddFeedFloat32(0, []int64{1, 3, 224, 224}, make([]float32, 150528))
	req.AddFetch(0)
	st := time.Now()
	for i := 0; i < 10; i++ {
		resp, err := cli.TorchPredict(req)
		if err != nil {
			t.Fatalf("failed to query torch model: %v", err)
		}
		fmt.Println(resp.GetTensorShape(0), resp.GetFloatVal(0))
	}
	fmt.Println("average response time : ", time.Since(st)/10)
}

func TestRequestPath(t *testing.T) {

	client := NewPredictClient(EndpointName, TestName)
	client.SetToken(TestToken)
	client.SetRequestPath("sleep")
	//client.SetEndpointType(EndpointTypeDirect)
	//client.SetIsInternalDirect(true)
	client.Init()
	req := "1"
	//client.AddHeader("headerName", "headerValue")
	resp, err := client.StringPredict(req)
	if err != nil {
		t.Fatalf(err.Error())
	} else {
		fmt.Printf("%v\n", resp)
	}
}

func TestCreateUrlWithQueryParams(t *testing.T) {
	client := NewPredictClient(EndpointName, "my_service")
	client.SetRequestPath("/infer")

	cases := []struct {
		name   string
		opts   []CallOption
		expect string
	}{
		{
			name:   "no options",
			opts:   nil,
			expect: "http://host.example.com/api/predict/my_service/infer",
		},
		{
			name: "with query params",
			opts: []CallOption{
				WithQueryParams(url.Values{"uid": []string{"1168XXXX"}}),
			},
			expect: "http://host.example.com/api/predict/my_service/infer?uid=1168XXXX",
		},
		{
			name: "with multiple query params",
			opts: []CallOption{
				WithQueryParam("uid", "1168XXXX"),
				WithQueryParam("debug", "1"),
			},
			expect: "http://host.example.com/api/predict/my_service/infer?debug=1&uid=1168XXXX",
		},
		{
			name: "with per-request path override",
			opts: []CallOption{
				WithRequestPath("sleep"),
			},
			expect: "http://host.example.com/api/predict/my_service/sleep",
		},
		{
			name: "with per-request path and query params",
			opts: []CallOption{
				WithRequestPath("/sleep"),
				WithQueryParam("uid", "1168XXXX"),
			},
			expect: "http://host.example.com/api/predict/my_service/sleep?uid=1168XXXX",
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			cfg := newCallConfig(c.opts)
			got := client.createUrl("host.example.com", cfg)
			if got != c.expect {
				t.Fatalf("createUrl mismatch, got: %s, expect: %s", got, c.expect)
			}
		})
	}
}

func TestQueryParams(t *testing.T) {
	client := NewPredictClient(EndpointName, TestName)
	client.SetToken(TestToken)
	client.Init()

	req := "1 2"
	params := url.Values{}
	params.Set("uid", "1168XXXX")
	resp, err := client.StringPredict(req, WithQueryParams(params))
	if err != nil {
		t.Fatalf(err.Error())
	} else {
		fmt.Printf("%v\n", resp)
	}
}

func TestRequestPathPerCall(t *testing.T) {
	client := NewPredictClient(EndpointName, TestName)
	client.SetToken(TestToken)
	client.Init()

	req := "1 2"
	resp, err := client.StringPredict(req, WithRequestPath("infer"))
	if err != nil {
		t.Fatalf(err.Error())
	} else {
		fmt.Printf("%v\n", resp)
	}
}
